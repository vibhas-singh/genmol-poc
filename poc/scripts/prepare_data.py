#!/usr/bin/env python
"""Prepare the Delaney (ESOL) dataset for GenMol fine-tuning.

Pipeline:
  1. Read the raw Delaney CSV and extract the `smiles` column.
  2. Standardize / canonicalize with RDKit and drop invalid or duplicate molecules.
  3. (Optional) apply light drug-like sanity filters (heavy-atom / MW bounds).
  4. Split into train / validation sets.
  5. Convert SMILES -> SAFE strings (the sequence format GenMol is trained on).
  6. Write train.safe / val.safe (one SAFE string per line, GenMol `UserDataset` format),
     the matching canonical-SMILES lists, and a dataset_stats.json summary.

Outputs are written to --out-dir and are consumed by finetune.py and generate_evaluate.py.
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import safe as sf
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from utils import (DRUG_LIKE_QED_MIN, DRUG_LIKE_SA_MAX, descriptor_summary,
                   molecule_descriptors, molecule_set_profile)

RDLogger.DisableLog("rdApp.*")


def murcko_scaffold(smiles):
    """Bemis-Murcko scaffold SMILES (empty string for acyclic / unparseable molecules)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(pairs, val_frac, seed):
    """Group molecules by Bemis-Murcko scaffold and assign whole groups to train/val.

    Largest scaffold groups fill the train set first; the remaining (rarer / singleton)
    scaffolds form the validation set, so train and val share no scaffold. This is a
    stricter generalization test than a random split.
    """
    scaffolds = {}
    for i, (smi, _) in enumerate(pairs):
        scaffolds.setdefault(murcko_scaffold(smi), []).append(i)
    # Sort groups largest-first; deterministic tie-break for reproducibility.
    groups = sorted(scaffolds.values(), key=lambda idxs: (len(idxs), -idxs[0]), reverse=True)
    n_train = len(pairs) - max(1, int(len(pairs) * val_frac))
    train_idx, val_idx = [], []
    for g in groups:
        if len(train_idx) + len(g) <= n_train:
            train_idx.extend(g)
        else:
            val_idx.extend(g)
    return ([pairs[i] for i in train_idx], [pairs[i] for i in val_idx], len(scaffolds))


def random_split(pairs, val_frac, seed):
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val], None


def canonicalize(smiles):
    """Return non-isomeric canonical SMILES (stereo stripped, matching SAFE) or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def smiles_to_safe(smiles):
    """Convert a canonical SMILES into a SAFE string (GenMol training format)."""
    try:
        return sf.SAFEConverter(ignore_stereo=True).encoder(smiles, allow_empty=True)
    except Exception:
        return None


OUTPUT_FILES = ["train.safe", "val.safe", "train_smiles.txt", "val_smiles.txt", "dataset_stats.json"]


def outputs_exist(out_dir):
    return all(os.path.exists(os.path.join(out_dir, f)) for f in OUTPUT_FILES)


def clean_and_filter(raw_smiles, min_heavy, max_heavy, max_mw):
    """Canonicalize -> dedupe -> light drug-like bounds. Returns (kept_smiles, drop_counts)."""
    canon, n_invalid = [], 0
    for s in raw_smiles:
        c = canonicalize(s.strip())
        if c is None:
            n_invalid += 1
        else:
            canon.append(c)

    seen, unique = set(), []
    for c in canon:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    kept = []
    for c in unique:
        d = molecule_descriptors(c)
        if d is None:
            continue
        if not (min_heavy <= d["heavy_atoms"] <= max_heavy) or d["mw"] > max_mw:
            continue
        kept.append(c)

    counts = {
        "invalid_dropped": n_invalid,
        "duplicates_dropped": len(canon) - len(unique),
        "descriptor_filtered_dropped": len(unique) - len(kept),
    }
    return kept, counts


def encode_to_safe(smiles_list):
    """SMILES -> SAFE, dropping anything the encoder cannot represent. Returns (pairs, n_fail)."""
    pairs, n_fail = [], 0
    for c in smiles_list:
        safe_str = smiles_to_safe(c)
        if safe_str:
            pairs.append((c, safe_str))
        else:
            n_fail += 1
    return pairs, n_fail


def write_split(out_dir, name, pairs):
    """Write <name>.safe (SAFE strings) and <name>_smiles.txt (canonical SMILES)."""
    with open(os.path.join(out_dir, f"{name}.safe"), "w") as f:
        f.writelines(s + "\n" for _, s in pairs)
    with open(os.path.join(out_dir, f"{name}_smiles.txt"), "w") as f:
        f.writelines(c + "\n" for c, _ in pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to delaney-processed.csv")
    parser.add_argument("--out-dir", required=True, help="Directory for prepared outputs")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--split", choices=["scaffold", "random"], default="scaffold",
                        help="Train/val split strategy. 'scaffold' (default) keeps whole "
                             "Bemis-Murcko scaffold groups disjoint between train and val; "
                             "'random' does a seeded uniform shuffle.")
    parser.add_argument("--min-heavy-atoms", type=int, default=5)
    parser.add_argument("--max-heavy-atoms", type=int, default=60)
    parser.add_argument("--max-mw", type=float, default=750.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if prepared outputs already exist (default: skip).")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Idempotent: reuse the frozen splits unless --force is given, so train/val never
    # change between fine-tuning and evaluation runs.
    if outputs_exist(args.out_dir) and not args.force:
        print(f"Prepared data already exists in {args.out_dir} (train.safe, val.safe, ...).")
        print("Reusing the existing splits. Pass --force to regenerate.")
        with open(os.path.join(args.out_dir, "dataset_stats.json")) as f:
            stats = json.load(f)
        print(json.dumps(stats, indent=2))
        return

    # Load the SMILES column.
    df = pd.read_csv(args.input)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column '{args.smiles_col}' not in {list(df.columns)}")
    raw_smiles = df[args.smiles_col].astype(str).tolist()

    # Clean/filter -> SAFE encode.
    kept, counts = clean_and_filter(raw_smiles, args.min_heavy_atoms, args.max_heavy_atoms, args.max_mw)
    safe_pairs, n_safe_fail = encode_to_safe(kept)

    # Split (scaffold by default) and write both splits.
    split_fn = scaffold_split if args.split == "scaffold" else random_split
    train_pairs, val_pairs, n_scaffolds = split_fn(safe_pairs, args.val_frac, args.seed)
    write_split(args.out_dir, "train", train_pairs)
    write_split(args.out_dir, "val", val_pairs)

    # Per-split stats (QED/SA means, avg token length, drug-like frac, descriptors) so
    # train/val can be compared and used as the reference the fine-tuned model moves toward.
    train_stats = molecule_set_profile(
        [c for c, _ in train_pairs], [s for _, s in train_pairs])
    val_stats = molecule_set_profile(
        [c for c, _ in val_pairs], [s for _, s in val_pairs])

    # Dataset statistics / description.
    stats = {
        "raw_records": len(raw_smiles),
        **counts,
        "safe_encoding_failed": n_safe_fail,
        "final_total": len(safe_pairs),
        "split_method": args.split,
        "num_scaffolds": n_scaffolds,
        "train": len(train_pairs),
        "val": len(val_pairs),
        "drug_like_definition": {
            "qed_min": DRUG_LIKE_QED_MIN,
            "sa_max": DRUG_LIKE_SA_MAX,
        },
        "descriptor_summary": descriptor_summary([c for c, _ in safe_pairs]),
        "train_stats": train_stats,
        "val_stats": val_stats,
    }
    with open(os.path.join(args.out_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))

    def fmt(value, spec):
        return format(value, spec) if value is not None else "n/a"

    for name, prof in (("TRAIN", train_stats), ("VAL", val_stats)):
        print(f"{name}: valid={prof['count']}  QED_mean={fmt(prof['qed_mean'], '.3f')}  "
              f"SA_mean={fmt(prof['sa_mean'], '.3f')}  "
              f"avg_tokens={fmt(prof['avg_token_length'], '.1f')}  "
              f"drug_like_frac={fmt(prof['drug_like_frac'], '.3f')}")
    print(f"\nWrote train.safe ({len(train_pairs)}) and val.safe ({len(val_pairs)}) to {args.out_dir}")


if __name__ == "__main__":
    main()
