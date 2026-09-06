#!/usr/bin/env python
"""Generate molecules from the base and fine-tuned GenMol models and compare them.

For each model we run de novo generation and compute:
  - validity    : fraction of requested samples that decode to a valid molecule
  - uniqueness  : fraction of valid samples that are unique
  - novelty     : fraction of unique valid samples NOT present in the training set
  - QED         : mean drug-likeness (RDKit)
  - SA score    : mean synthetic accessibility (TDC 'sa' oracle; lower is easier)
  - diversity   : internal Tanimoto diversity (TDC evaluator)
  - drug-like %  : fraction passing QED>=0.6 and SA<=4 ("quality", GenMol convention)
  - similarity  : mean nearest-neighbour Tanimoto (ECFP4) to the training set
                  -> measures alignment to the target chemical domain
    - FCD         : Frechet ChemNet Distance to the selected reference (lower = closer;
                                    needs fcd_torch)

It writes per-model CSVs, a comparison JSON/CSV, five example molecules (SMILES, QED, SA,
nearest-reference similarity), a labelled molecule-grid PNG, and a train/base/fine-tuned
property-distribution PNG (QED/MW/logP). The reference is the training set by default and can
be replaced with a held-out set through --ref-smiles.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED
from tdc import Evaluator

from genmol.sampler import Sampler
from genmol.utils.bracket_safe_converter import safe2bracketsafe
from utils import (DRUG_LIKE_QED_MIN, DRUG_LIKE_SA_MAX, get_sa_oracle,
                   molecule_descriptors)

RDLogger.DisableLog("rdApp.*")


def resolve_run_checkpoints(checkpoint_dir):
    """Return first, best, and last checkpoints from a fine-tuning run directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if (checkpoint_dir / "checkpoints").is_dir():
        checkpoint_dir = checkpoint_dir / "checkpoints"
    best = checkpoint_dir / "best.ckpt"
    if not best.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {best}")

    periodic = []
    for path in checkpoint_dir.glob("*.ckpt"):
        if path.name == "best.ckpt":
            continue
        numbers = re.findall(r"\d+", path.stem)
        if numbers:
            periodic.append((int(numbers[-1]), path))
    if not periodic:
        raise FileNotFoundError(
            f"No numbered periodic checkpoints found in {checkpoint_dir}")

    periodic.sort(key=lambda item: item[0])
    return {
        "first": str(periodic[0][1]),
        "best": str(best),
        "last": str(periodic[-1][1]),
    }


def seed_everything(seed):
    """Seed Python/NumPy/torch so de novo sampling is reproducible across runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def morgan(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def nearest_train_similarity(gen_smiles, train_fps):
    """Max Tanimoto (ECFP4) between each generated molecule and the training set."""
    sims = []
    for s in gen_smiles:
        fp = morgan(s)
        if fp is None or not train_fps:
            sims.append(np.nan)
            continue
        sims.append(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
    return sims


def canonical_set(smiles_list):
    out = set()
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            out.add(Chem.MolToSmiles(mol, canonical=True))
    return out


def resolve_gen_kwargs(args, use_domain_lengths=False):
    """Pick de novo sampling settings: V2 (default) or V1 defaults, with per-arg overrides.

    Values mirror the repo's own configs (denovo/hparams.yaml, hparams_v2.yaml).
    """
    base = dict(softmax_temp=1.0, randomness=0.3, min_add_len=60) if args.v2 \
        else dict(softmax_temp=0.5, randomness=0.5, min_add_len=40)
    if use_domain_lengths:
        base["min_add_len"] = 1
    return dict(
        softmax_temp=args.softmax_temp if args.softmax_temp is not None else base["softmax_temp"],
        randomness=args.randomness if args.randomness is not None else base["randomness"],
        min_add_len=args.min_add_len if args.min_add_len is not None else base["min_add_len"],
    )


class DeviceSampler(Sampler):
    """GenMol sampler that explicitly places generation state on the requested device."""

    def __init__(self, checkpoint_path, device="auto"):
        super().__init__(checkpoint_path)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA generation requested, but torch.cuda.is_available() is false")
        self.model.to(torch.device(device))
        self.mdlm.to_device(self.model.device)
        self.model.backbone.eval()


class DomainLengthSampler(DeviceSampler):
    """Sample generation lengths from a SAFE dataset instead of pretrained len.pk."""

    def __init__(self, checkpoint_path, safe_path, length_seed, device="auto"):
        super().__init__(checkpoint_path, device=device)
        with open(safe_path) as f:
            sequences = [line.strip() for line in f if line.strip()]
        if self.model.config.training.get("use_bracket_safe"):
            sequences = [safe2bracketsafe(sequence) for sequence in sequences]
        encoded = self.model.tokenizer(sequences, truncation=True,
                                       max_length=self.model.config.model.max_position_embeddings)
        self.domain_sequence_lengths = [len(ids) for ids in encoded["input_ids"]]
        if not self.domain_sequence_lengths:
            raise ValueError(f"No SAFE sequences found in length reference: {safe_path}")
        self.length_rng = random.Random(length_seed)

    def _insert_mask(self, x, num_samples, min_add_len=1, **kwargs):
        x = x[0]
        generated = []
        for _ in range(num_samples):
            sampled_length = self.length_rng.choice(self.domain_sequence_lengths)
            add_seq_len = max(sampled_length - len(x), min_add_len)
            generated.append(torch.hstack([
                x[:-1],
                torch.full((add_seq_len,), self.model.mask_index),
                x[-1:],
            ]))
        pad_len = max(len(sequence) for sequence in generated)
        generated = [torch.hstack([
            sequence,
            torch.full((pad_len - len(sequence),), self.pad_index),
        ]) for sequence in generated]
        return torch.stack(generated)


def training_min_add_len(checkpoint_path, safe_path, device="auto"):
    """Return the shortest training sequence as a number of inserted mask tokens."""
    sampler = DeviceSampler(checkpoint_path, device=device)
    with open(safe_path) as safe_file:
        sequences = [line.strip() for line in safe_file if line.strip()]
    if not sequences:
        raise ValueError(f"No SAFE sequences found in length reference: {safe_path}")
    if sampler.model.config.training.get("use_bracket_safe"):
        sequences = [safe2bracketsafe(sequence) for sequence in sequences]
    encoded = sampler.model.tokenizer(
        sequences,
        truncation=True,
        max_length=sampler.model.config.model.max_position_embeddings,
    )
    minimum = max(min(len(input_ids) for input_ids in encoded["input_ids"]) - 2, 1)
    del sampler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return minimum


def load_training_reference(train_smiles_path):
    """Load training set as (SMILES list, canonical SMILES set, ECFP4 fps) for novelty/similarity/FCD."""
    with open(train_smiles_path) as f:
        smiles = [ln.strip() for ln in f if ln.strip()]
    train_canon = canonical_set(smiles)
    train_fps = [fp for fp in (morgan(s) for s in train_canon) if fp is not None]
    return smiles, train_canon, train_fps


def select_examples(finetuned_df, n=5):
    """Return up to n novel, drug-like generations, ranked by QED.

    Drug-like uses the same gate as the aggregate metric: QED >= 0.6 and SA <= 4.
    A short result is preferable to silently presenting non-qualifying molecules.
    """
    qualifying = finetuned_df[
        finetuned_df["novel"]
        & (finetuned_df["qed"] >= DRUG_LIKE_QED_MIN)
        & (finetuned_df["sa"] <= DRUG_LIKE_SA_MAX)
    ]
    if len(qualifying) < n:
        print(f"[warn] only {len(qualifying)} of {n} requested novel drug-like examples "
              f"met QED >= {DRUG_LIKE_QED_MIN} and SA <= {DRUG_LIKE_SA_MAX}")
    return qualifying.sort_values("qed", ascending=False).head(n)[
        ["smiles", "qed", "sa", "nearest_train_sim"]
    ]


def _descriptor_frame(smiles_list):
    """QED / MW / logP for a list of SMILES (for distribution plots)."""
    rows = [molecule_descriptors(s) for s in smiles_list]
    return pd.DataFrame([{"qed": r["qed"], "mw": r["mw"], "logp": r["logp"]}
                         for r in rows if r is not None])


def draw_examples_grid(examples_df, out_path):
    """Render the five showcase molecules as a labelled PNG grid (RDKit)."""
    try:
        from rdkit.Chem import Draw
    except Exception as e:
        print(f"[skip] molecule grid: {e}")
        return
    mols, legends = [], []
    for _, r in examples_df.iterrows():
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            continue
        mols.append(mol)
        legends.append(f"QED {r['qed']:.2f} | SA {r['sa']:.2f} | sim {r['nearest_train_sim']:.2f}")
    if not mols:
        print("[skip] molecule grid: no valid molecules")
        return
    img = Draw.MolsToGridImage(mols, molsPerRow=min(5, len(mols)), subImgSize=(300, 250), legends=legends)
    img.save(out_path)
    print(f"Saved molecule grid -> {out_path}")


def plot_property_distributions(train_smiles, base_smiles, ft_smiles, out_path):
    """Overlay QED/MW/logP histograms for train vs base vs fine-tuned (domain-shift evidence)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[skip] distribution plots: {e}")
        return
    frames = {name: _descriptor_frame(s) for name, s in
              {"train": train_smiles, "base": base_smiles, "finetuned": ft_smiles}.items()}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, title) in zip(axes, [("qed", "QED"), ("mw", "MW"), ("logp", "logP")]):
        for name, frame in frames.items():
            if len(frame):
                ax.hist(frame[col], bins=30, density=True, histtype="step", linewidth=2, label=name)
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Property distributions: train vs base vs fine-tuned")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved property distribution plot -> {out_path}")


def compute_fcd(train_smiles, base_smiles, ft_smiles):
    """Frechet ChemNet Distance to the selected reference (lower is closer)."""
    try:
        import torch
        from fcd_torch import FCD
    except Exception as e:
        print(f"[skip] FCD (pip install fcd_torch): {e}")
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"FCD device: {device}")
    fcd = FCD(device=device, n_jobs=1)
    ref = [s for s in train_smiles if Chem.MolFromSmiles(s) is not None]
    scores = {}
    for name, gen in {"base": base_smiles, "finetuned": ft_smiles}.items():
        valid = [s for s in gen if Chem.MolFromSmiles(s) is not None]
        scores[name] = float(fcd(ref, valid)) if valid else None
    return scores


def evaluate_model(name, ckpt, num_samples, gen_kwargs, train_canon, ref_fps,
                   oracle_sa, diversity_eval, out_dir, seed, length_safe=None, device="auto"):
    print(f"\n=== {name}: sampling {num_samples} molecules from {ckpt} ===")
    seed_everything(seed)
    sampler = DomainLengthSampler(ckpt, length_safe, seed, device=device) if length_safe \
        else DeviceSampler(ckpt, device=device)
    generation_device = str(next(sampler.model.parameters()).device)
    print(f"Generation device: {generation_device}")
    samples = sampler.de_novo_generation(num_samples=num_samples, **gen_kwargs)

    # Parse -> canonicalize -> dedupe, then score the unique set.
    valid = [s for s in samples if s and Chem.MolFromSmiles(s) is not None]
    canon = [Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True) for s in valid]
    unique = sorted(set(canon))

    qed = [QED.qed(Chem.MolFromSmiles(s)) for s in unique]
    sa = list(oracle_sa(unique)) if unique else []
    # Similarity/FCD use the reference set (train by default; val for unbiased sweep selection).
    sims = nearest_train_similarity(unique, ref_fps)
    novel_flags = [c not in train_canon for c in unique]

    df = pd.DataFrame({
        "smiles": unique,
        "qed": qed,
        "sa": sa,
        "nearest_train_sim": sims,
        "novel": novel_flags,
    })
    df.to_csv(os.path.join(out_dir, f"{name}_generated.csv"), index=False)

    # "quality" = novelty-agnostic drug-likeness gate used by GenMol (QED>=0.6 & SA<=4).
    drug_like = df[
        (df["qed"] >= DRUG_LIKE_QED_MIN) & (df["sa"] <= DRUG_LIKE_SA_MAX)
    ]
    metrics = {
        "model": name,
        "checkpoint": ckpt,
        "requested": num_samples,
        "validity": len(valid) / num_samples if num_samples else 0.0,
        "uniqueness": len(unique) / len(valid) if valid else 0.0,
        "novelty": float(np.mean(novel_flags)) if novel_flags else 0.0,
        "qed_mean": float(np.mean(qed)) if qed else 0.0,
        "sa_mean": float(np.mean(sa)) if sa else 0.0,
        "diversity": float(diversity_eval(unique)) if len(unique) > 1 else 0.0,
        "drug_like_frac": len(drug_like) / num_samples if num_samples else 0.0,
        "nearest_train_sim_mean": float(np.nanmean(sims)) if sims else 0.0,
        "length_prior": length_safe or "pretrained_len.pk",
        "min_add_len": gen_kwargs["min_add_len"],
        "generation_device": generation_device,
    }
    print(json.dumps(metrics, indent=2))
    return metrics, df


def write_comparison_artifacts(out_dir, train_smiles, ref_smiles,
                               base_metrics, base_df, finetuned_metrics, finetuned_df):
    """Write metrics, examples, and plots for one base-versus-checkpoint comparison."""
    os.makedirs(out_dir, exist_ok=True)
    base_metrics = dict(base_metrics)
    finetuned_metrics = dict(finetuned_metrics)
    base_df.to_csv(os.path.join(out_dir, "base_generated.csv"), index=False)
    finetuned_df.to_csv(os.path.join(out_dir, "finetuned_generated.csv"), index=False)

    fcd_scores = compute_fcd(
        ref_smiles, base_df["smiles"].tolist(), finetuned_df["smiles"].tolist())
    if fcd_scores:
        base_metrics["fcd"] = fcd_scores["base"]
        finetuned_metrics["fcd"] = fcd_scores["finetuned"]

    comparison = pd.DataFrame([base_metrics, finetuned_metrics])
    comparison.to_csv(os.path.join(out_dir, "comparison.csv"), index=False)
    with open(os.path.join(out_dir, "comparison.json"), "w") as comparison_file:
        json.dump({"base": base_metrics, "finetuned": finetuned_metrics},
                  comparison_file, indent=2)

    examples = select_examples(finetuned_df, n=5)
    examples.to_csv(os.path.join(out_dir, "five_examples.csv"), index=False)
    draw_examples_grid(examples, os.path.join(out_dir, "five_examples.png"))
    plot_property_distributions(
        train_smiles,
        base_df["smiles"].tolist(),
        finetuned_df["smiles"].tolist(),
        os.path.join(out_dir, "property_distributions.png"),
    )

    print("\n=== Base vs Fine-tuned ===")
    print(comparison.to_string(index=False))
    print("\n=== Five example molecules (fine-tuned) ===")
    print(examples.to_string(index=False))
    return base_metrics, finetuned_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ckpt", required=True, help="Pretrained checkpoint (model_v2.ckpt by default, or model.ckpt for V1)")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--finetuned-ckpt", help="One fine-tuned checkpoint")
    checkpoint_group.add_argument(
        "--checkpoint-dir",
        help="Run or checkpoints directory; evaluates its first, best, and last checkpoints",
    )
    parser.add_argument("--train-smiles", required=True, help="train_smiles.txt from prepare_data.py (novelty reference)")
    parser.add_argument("--ref-smiles", default=None,
                        help="Reference set for FCD + nearest-neighbour similarity (default: --train-smiles). "
                             "Pass val_smiles.txt for an unbiased, held-out domain reference (recommended "
                             "for sweep selection to avoid rewarding memorization of the training set).")
    parser.add_argument("--length-source", choices=("pretrained", "train"), default="train",
                        help="Sequence-length prior: GenMol's pretrained len.pk or token lengths "
                             "from the training SAFE file [default: train].")
    parser.add_argument("--length-safe", default=None,
                        help="Training SAFE file used for the train length prior and to derive "
                             "the min_train floor (default: train.safe next to --train-smiles).")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--v2", action=argparse.BooleanOptionalAction, default=True,
                        help="Use V2 (extended/bracket SAFE) de novo sampling defaults [default] "
                             "(softmax_temp=1.0, randomness=0.3, min_add_len=60). Pass --no-v2 for "
                             "V1 defaults (0.5/0.5/40). Decoding is auto-selected from each "
                             "checkpoint's own config.")
    parser.add_argument("--softmax-temp", type=float, default=None)
    parser.add_argument("--randomness", type=float, default=None)
    parser.add_argument("--min-add-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible sampling.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="Generation device. 'auto' uses CUDA when available [default].")
    args = parser.parse_args()

    if args.checkpoint_dir and args.min_add_len is not None:
        parser.error("--min-add-len is only available with --finetuned-ckpt; folder mode "
                     "always evaluates min_60 and min_train")

    seed_everything(args.seed)
    print(f"Requested generation device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("RDKit/TDC metric device: cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    # Novelty is always vs the training set; FCD/similarity use --ref-smiles (default: train).
    train_smiles, train_canon, train_fps = load_training_reference(args.train_smiles)
    if args.ref_smiles:
        ref_smiles, _, ref_fps = load_training_reference(args.ref_smiles)
    else:
        ref_smiles, ref_fps = train_smiles, train_fps

    inferred_length_safe = os.path.join(os.path.dirname(args.train_smiles), "train.safe")
    training_safe = args.length_safe or inferred_length_safe
    if args.length_source == "train" and not os.path.exists(training_safe):
        raise FileNotFoundError(f"Training length-prior SAFE file not found: {training_safe}")
    length_safe = training_safe if args.length_source == "train" else None
    gen_kwargs = resolve_gen_kwargs(args, use_domain_lengths=bool(length_safe))
    length_source = length_safe or "GenMol pretrained len.pk"
    print(f"Generation length prior: {length_source} (min_add_len={gen_kwargs['min_add_len']})")

    oracle_sa = get_sa_oracle()
    diversity_eval = Evaluator("diversity")

    if args.finetuned_ckpt:
        base_m, base_df = evaluate_model(
            "base", args.base_ckpt, args.num_samples, gen_kwargs, train_canon, ref_fps,
            oracle_sa, diversity_eval, args.out_dir, args.seed, length_safe, args.device)
        ft_m, ft_df = evaluate_model(
            "finetuned", args.finetuned_ckpt, args.num_samples, gen_kwargs, train_canon,
            ref_fps, oracle_sa, diversity_eval, args.out_dir, args.seed, length_safe, args.device)
        write_comparison_artifacts(
            args.out_dir, train_smiles, ref_smiles, base_m, base_df, ft_m, ft_df)
        return

    if not os.path.exists(training_safe):
        raise FileNotFoundError(f"Training SAFE file not found for min_train: {training_safe}")
    checkpoints = resolve_run_checkpoints(args.checkpoint_dir)
    train_min = training_min_add_len(args.base_ckpt, training_safe, args.device)
    configurations = {"min_60": 60, "min_train": train_min}
    print(f"Checkpoint matrix: {', '.join(checkpoints)} x {', '.join(configurations)}")
    print(f"Training-derived minimum inserted length: {train_min}")

    aggregate_rows = []
    for config_name, minimum in configurations.items():
        config_kwargs = dict(gen_kwargs, min_add_len=minimum)
        first_out_dir = os.path.join(args.out_dir, "first", config_name)
        os.makedirs(first_out_dir, exist_ok=True)
        base_m, base_df = evaluate_model(
            "base", args.base_ckpt, args.num_samples, config_kwargs, train_canon, ref_fps,
            oracle_sa, diversity_eval, first_out_dir, args.seed, length_safe, args.device)
        base_row = {
            "checkpoint_role": "base",
            "length_config": config_name,
            **base_m,
        }
        aggregate_rows.append(base_row)

        for checkpoint_name, checkpoint_path in checkpoints.items():
            leaf_out_dir = os.path.join(args.out_dir, checkpoint_name, config_name)
            os.makedirs(leaf_out_dir, exist_ok=True)
            ft_m, ft_df = evaluate_model(
                "finetuned", checkpoint_path, args.num_samples, config_kwargs, train_canon,
                ref_fps, oracle_sa, diversity_eval, leaf_out_dir, args.seed,
                length_safe, args.device)
            enriched_base_m, enriched_ft_m = write_comparison_artifacts(
                leaf_out_dir, train_smiles, ref_smiles, base_m, base_df, ft_m, ft_df)
            base_row.update(enriched_base_m)
            aggregate_rows.append({
                "checkpoint_role": checkpoint_name,
                "length_config": config_name,
                **enriched_ft_m,
            })

    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(os.path.join(args.out_dir, "checkpoint_comparison.csv"), index=False)
    with open(os.path.join(args.out_dir, "checkpoint_comparison.json"), "w") as aggregate_file:
        json.dump(aggregate_rows, aggregate_file, indent=2)
    print("\n=== Checkpoint matrix summary ===")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
