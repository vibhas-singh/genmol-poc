#!/usr/bin/env python
"""Aggregate a GenMol hyperparameter sweep and select the best model.

Scans <sweep-dir>/*/eval/comparison.json (written by generate_evaluate.py) plus each run's
optional val_loss.txt, ranks the runs, and writes a sorted sweep_results.csv.

Selection philosophy (matches the PoC's "is it improving?" criteria):
  - GATE on generation health AND improvement over base: validity/novelty must clear absolute
    thresholds, AND the fine-tuned model must not be worse than the pretrained base on validity,
    novelty, or drug-likeness (beyond --degrade-tol). This prevents selecting a run that is worse
    than the base model on every relevant measure.
  - Among gated runs, rank by a composite of drug-likeness + novelty + diversity, with a penalty
    for near-duplicate generations (high nearest-ref similarity = memorization) and a small bonus
    for lower FCD. Ties are broken by LOWER val_loss (noisy, so only a tie-breaker).
  - Base metrics and base->finetuned deltas are recorded per run for transparency. Run
    generate_evaluate.py with --ref-smiles val_smiles.txt so FCD/similarity use a held-out
    reference, keeping selection from rewarding memorization of the training set.
"""

import argparse
import glob
import json
import os
import re

import pandas as pd

METRIC_COLS = ["validity", "uniqueness", "novelty", "qed_mean", "sa_mean",
               "diversity", "drug_like_frac", "nearest_train_sim_mean", "fcd"]


def parse_tag(name):
    """Extract (freeze, lr) from a run dir named like 'f6_lr1e-4'."""
    m = re.match(r"f(\d+)_lr([0-9.eE+-]+)", name)
    return (int(m.group(1)), float(m.group(2))) if m else (None, None)


def load_run(run_dir):
    comparison = os.path.join(run_dir, "eval", "comparison.json")
    if not os.path.exists(comparison):
        return None
    data = json.load(open(comparison))
    ft = data.get("finetuned", {})
    base = data.get("base", {})
    freeze, lr = parse_tag(os.path.basename(run_dir))
    row = {"run": os.path.basename(run_dir), "freeze_layers": freeze, "lr": lr}
    row.update({c: ft.get(c) for c in METRIC_COLS})

    # Base metrics + base->finetuned deltas, so selection can require improvement over the
    # pretrained model (not just a high absolute score).
    def delta(a, b):
        return (a - b) if (a is not None and b is not None) else None
    row["base_validity"] = base.get("validity")
    row["base_novelty"] = base.get("novelty")
    row["base_drug_like_frac"] = base.get("drug_like_frac")
    row["base_fcd"] = base.get("fcd")
    row["d_validity"] = delta(ft.get("validity"), base.get("validity"))
    row["d_novelty"] = delta(ft.get("novelty"), base.get("novelty"))
    row["d_drug_like"] = delta(ft.get("drug_like_frac"), base.get("drug_like_frac"))
    row["d_fcd"] = delta(base.get("fcd"), ft.get("fcd"))   # positive = fine-tuned is closer

    # Prefer the richer run_summary.json (val_loss, early-stop step, wall time); fall back to val_loss.txt.
    summary_path = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(summary_path):
        s = json.load(open(summary_path))
        row["val_loss"] = s.get("best_val_loss")
        row["baseline_val_loss"] = s.get("baseline_val_loss")
        row["stopped_early"] = s.get("stopped_early")
        row["stop_epoch"] = s.get("stop_epoch")
        row["stop_step"] = s.get("stop_step")
        row["wall_time_sec"] = s.get("wall_time_sec")
    else:
        vl_path = os.path.join(run_dir, "val_loss.txt")
        txt = open(vl_path).read().strip() if os.path.exists(vl_path) else ""
        row["val_loss"] = float(txt) if txt else None
    return row


def composite_score(row):
    """Rank healthy PoC runs with a transparent, heuristic score.

    The weights balance metrics with different roles rather than representing a fitted objective:
    drug-likeness is primary, novelty and diversity preserve useful coverage, similarity above
    0.8 is treated as a near-copy warning, and FCD receives a bounded secondary contribution.
    Tune these constants when the project has a domain-specific utility function.
    """
    dl = row.get("drug_like_frac") or 0.0
    nov = row.get("novelty") or 0.0
    div = row.get("diversity") or 0.0
    sim = row.get("nearest_train_sim_mean") or 0.0
    fcd = row.get("fcd")
    memorization_penalty = max(0.0, sim - 0.8)     # discourage near-duplicates of training
    fcd_bonus = (1.0 / (1.0 + fcd)) if fcd is not None else 0.0
    return dl + 0.5 * nov + 0.25 * div - memorization_penalty + 0.25 * fcd_bonus


def non_degraded(row, tol):
    """True if fine-tuned is not worse than base on validity/novelty/drug-likeness (beyond tol)."""
    for key in ("d_validity", "d_novelty", "d_drug_like"):
        d = row.get(key)
        if d is not None and d < -tol:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", help="Directory containing the per-run sweep folders")
    parser.add_argument("--min-validity", type=float, default=0.9)
    parser.add_argument("--min-novelty", type=float, default=0.8)
    parser.add_argument("--degrade-tol", type=float, default=0.02,
                        help="Allowed slack when requiring non-degradation vs base (metric noise margin).")
    args = parser.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.sweep_dir, "*"))):
        if os.path.isdir(d):
            row = load_run(d)
            if row:
                rows.append(row)
    if not rows:
        print(f"No runs with eval/comparison.json found under {args.sweep_dir}")
        return

    df = pd.DataFrame(rows)
    df["score"] = df.apply(composite_score, axis=1)
    df["non_degraded"] = df.apply(lambda r: non_degraded(r, args.degrade_tol), axis=1)
    # Gate: absolute health (validity/novelty) AND not worse than the base model.
    df["passes_gate"] = ((df["validity"].fillna(0) >= args.min_validity) &
                         (df["novelty"].fillna(0) >= args.min_novelty) &
                         df["non_degraded"])
    # Rank: gate, then composite score, then LOWER val_loss as the documented tie-break.
    df["_vl"] = df["val_loss"].fillna(float("inf"))
    df = (df.sort_values(["passes_gate", "score", "_vl"], ascending=[False, False, True])
            .drop(columns="_vl").reset_index(drop=True))

    out_csv = os.path.join(args.sweep_dir, "sweep_results.csv")
    df.to_csv(out_csv, index=False)

    print(df.to_string(index=False))
    print(f"\nWrote {out_csv}")

    gated = df[df["passes_gate"]]
    if len(gated):
        best = gated.iloc[0]
        ckpt = os.path.join(args.sweep_dir, best["run"], "checkpoints", "best.ckpt")
        print(f"\nSelected best: {best['run']}  (score={best['score']:.3f}, "
              f"drug_like={best.get('drug_like_frac')}, novelty={best.get('novelty')}, "
              f"Δdrug_like={best.get('d_drug_like')}, val_loss={best.get('val_loss')})")
        print(f"Checkpoint: {ckpt}")
    else:
        print("\nNo run passed the gates (absolute health + non-degradation vs base) — "
              "inspect sweep_results.csv; relax --min-validity/--min-novelty/--degrade-tol if needed.")


if __name__ == "__main__":
    main()
