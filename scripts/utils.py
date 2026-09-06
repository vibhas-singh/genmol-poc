"""Shared utilities for PoC training and evaluation scripts."""

import csv
import os
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED

# Descriptors summarized for every molecule set (data splits and generated samples).
DESCRIPTOR_KEYS = ["mw", "logp", "qed", "hbd", "hba", "rings", "heavy_atoms"]
SAFE_MAX_TOKENS = 256
DRUG_LIKE_QED_MIN = 0.6
DRUG_LIKE_SA_MAX = 4.0


def molecule_descriptors(smiles):
    """RDKit descriptors for one SMILES (mw/logp/qed/hbd/hba/rings/heavy_atoms), or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "qed": QED.qed(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rings": Descriptors.RingCount(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
    }


def summarize(values):
    """mean/std/min/max/count over finite values (sample std, matching pandas.std())."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    return {"mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max()), "count": int(arr.size)}


def descriptor_summary(smiles_list):
    """Per-descriptor mean/std/min/max over a list of SMILES (the dataset description)."""
    rows = [d for d in (molecule_descriptors(s) for s in smiles_list) if d is not None]
    return {key: {stat: summarize([r[key] for r in rows])[stat]
                  for stat in ("mean", "std", "min", "max")}
            for key in DESCRIPTOR_KEYS}


@lru_cache(maxsize=1)
def get_sa_oracle():
    """Return the process-wide TDC synthetic-accessibility oracle."""
    from tdc import Oracle
    return Oracle("sa")


def sa_scores(smiles_list, oracle=None):
    """SA score per molecule (lower = easier). [] if the TDC 'sa' oracle is unavailable."""
    if not smiles_list:
        return []
    if oracle is None:
        try:
            oracle = get_sa_oracle()
        except Exception as exc:
            print(f"[warn] SA scores skipped (TDC 'sa' oracle unavailable): {exc}")
            return []
    return [float(s) for s in oracle(list(smiles_list))]


def drug_like_fraction(qeds, sas, qed_threshold=DRUG_LIKE_QED_MIN,
                       sa_threshold=DRUG_LIKE_SA_MAX):
    """Fraction with QED >= threshold AND SA <= threshold (GenMol 'quality' gate)."""
    if not qeds or not sas or len(qeds) != len(sas):
        return 0.0
    passed = sum(1 for q, s in zip(qeds, sas) if q >= qed_threshold and s <= sa_threshold)
    return passed / len(qeds)


@lru_cache(maxsize=1)
def load_safe_tokenizer():
    """Return the process-wide SAFE-GPT tokenizer used by GenMol."""
    from safe.tokenizer import SAFETokenizer
    return SAFETokenizer.from_pretrained("datamol-io/safe-gpt").get_pretrained()


def safe_token_lengths(safe_list, tokenizer=None, max_length=SAFE_MAX_TOKENS):
    """Tokenized length of each SAFE string. [] if a tokenizer cannot be loaded."""
    if not safe_list:
        return []
    if tokenizer is None:
        try:
            tokenizer = load_safe_tokenizer()
        except Exception as exc:
            print(f"[warn] token lengths skipped (SAFE tokenizer unavailable): {exc}")
            return []
    encoded = tokenizer(list(safe_list), truncation=True, max_length=max_length)
    return [len(ids) for ids in encoded["input_ids"]]


def molecule_set_profile(smiles_list, safe_list):
    """Summarize descriptors, SA, SAFE length, and drug-like fraction for one split."""
    qeds = [descriptor["qed"] for descriptor in
            (molecule_descriptors(smiles) for smiles in smiles_list)
            if descriptor is not None]
    sas = sa_scores(smiles_list)
    lengths = safe_token_lengths(safe_list)
    return {
        "count": len(smiles_list),
        "qed_mean": summarize(qeds)["mean"],
        "sa_mean": summarize(sas)["mean"] if sas else None,
        "avg_token_length": summarize(lengths)["mean"] if lengths else None,
        "drug_like_frac": drug_like_fraction(qeds, sas) if sas else None,
        "descriptor_summary": descriptor_summary(smiles_list),
    }


def loss_axis_limits(scale_losses, protected_losses):
    """Return robust y-limits that always include validation and baseline losses."""
    lower, upper = np.percentile(scale_losses, [1, 99])
    if protected_losses:
        lower = min(lower, min(protected_losses))
        upper = max(upper, max(protected_losses))
    span = upper - lower
    padding = max(span * 0.1, abs(upper) * 0.02, 1e-6)
    return max(0, lower - padding), upper + padding


def plot_loss_curves(metrics_path, out_dir, baseline_val=None):
    """Plot step-level training loss and validation loss from Lightning's CSV log."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(metrics_path):
        print(f"[warn] loss curves skipped: metrics file not found at {metrics_path}")
        return None

    train_points = []
    val_points = []
    with open(metrics_path, newline="") as metrics_file:
        for row in csv.DictReader(metrics_file):
            step = row.get("step")
            if step in (None, ""):
                continue
            if row.get("train_loss") not in (None, ""):
                loss = float(row["train_loss"])
                if np.isfinite(loss):
                    train_points.append((int(float(step)), loss))
            if row.get("val_loss") not in (None, ""):
                loss = float(row["val_loss"])
                if np.isfinite(loss):
                    epoch = int(float(row["epoch"])) + 1 if row.get("epoch") not in (None, "") else None
                    val_points.append((int(float(step)), loss, epoch))

    if not train_points and not val_points:
        print(f"[warn] loss curves skipped: no loss values found in {metrics_path}")
        return None

    train_points.sort()
    val_points.sort()
    figure, axis = plt.subplots(figsize=(9, 5))
    scale_losses = []
    protected_losses = []
    if train_points:
        train_steps, train_losses = zip(*train_points)
        axis.plot(train_steps, train_losses, color="tab:blue", alpha=0.2,
                  linewidth=1, label="Training loss (raw)")
        window = min(20, len(train_losses))
        if window > 1:
            smoothed = np.convolve(train_losses, np.ones(window) / window, mode="valid")
            smooth_steps = train_steps[window - 1:]
            axis.plot(smooth_steps, smoothed, color="tab:blue", linewidth=2,
                      label=f"Training loss ({window}-step mean)")
            scale_losses.extend(smoothed)
        else:
            scale_losses.extend(train_losses)
    if val_points:
        val_steps, val_losses, val_epochs = zip(*val_points)
        axis.plot(val_steps, val_losses, color="tab:orange", marker="o",
                  linewidth=2, label="Validation loss")
        label_every = max(1, len(val_points) // 10)
        for index, (step, loss, epoch) in enumerate(val_points):
            if epoch is not None and (index % label_every == 0 or index == len(val_points) - 1):
                axis.annotate(f"E{epoch}", (step, loss), xytext=(0, 7),
                              textcoords="offset points", ha="center", fontsize=8)
        scale_losses.extend(val_losses)
        protected_losses.extend(val_losses)
    if baseline_val is not None:
        axis.scatter(0, baseline_val, color="tab:gray", zorder=3,
                     label="Pretrained validation baseline")
        scale_losses.append(baseline_val)
        protected_losses.append(baseline_val)
    if val_points:
        checkpoint_points = [point for point in val_points if point[0] > 0] or val_points
        best_step, best_val, best_epoch = min(checkpoint_points, key=lambda point: point[1])
        axis.scatter(best_step, best_val, marker="*", s=180, color="tab:green",
                     edgecolor="black", linewidth=0.5, zorder=4,
                     label=f"Best val ({best_val:.4f})")
        epoch_label = f"epoch {best_epoch}, " if best_epoch is not None else ""
        axis.annotate(f"Best: {epoch_label}{best_val:.4f}", (best_step, best_val),
                      xytext=(10, -18), textcoords="offset points", fontsize=9)

    if len(scale_losses) > 1:
        axis.set_ylim(*loss_axis_limits(scale_losses, protected_losses))

    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Loss")
    axis.set_title("Fine-tuning loss curves")
    axis.grid(alpha=0.25)
    axis.margins(x=0.02, y=0.08)
    axis.legend()
    figure.tight_layout()
    output_path = os.path.join(out_dir, "loss_curves.png")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Loss curves -> {output_path}")
    return output_path