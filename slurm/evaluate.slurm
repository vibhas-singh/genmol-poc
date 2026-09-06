#!/bin/bash
#SBATCH --job-name=genmol-evaluate
#SBATCH --partition=req-576              # <-- adjust to your cluster's A100 partition
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err

# GenMol PoC — first/best/last checkpoint comparison (1 A100). Run after finetune.slurm.
# NOTE: submit from the workspace root so the logs/ dir in --output/--error exists.
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate genmol

WORKSPACE="${WORKSPACE:-$SLURM_SUBMIT_DIR}"
GENMOL_ROOT="$WORKSPACE/genmol"
POC="$WORKSPACE/poc"
DATA_DIR="$POC/outputs/data"
# V2 by default (uses model_v2.ckpt + V2 sampling defaults). For V1, set MODEL_VERSION=v1.
MODEL_VERSION="${MODEL_VERSION:-v2}"
if [ "$MODEL_VERSION" = "v1" ]; then
    PRETRAINED_CKPT="${PRETRAINED_CKPT:-$WORKSPACE/models/genmol_v1_v1.0/model.ckpt}"
    V2_FLAG="--no-v2"
else
    PRETRAINED_CKPT="${PRETRAINED_CKPT:-$WORKSPACE/models/genmol_v2_v1.0/model_v2.ckpt}"
    V2_FLAG="--v2"
fi
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: pretrained checkpoint not found: $PRETRAINED_CKPT" >&2
    exit 1
fi

# Evaluate an explicit FINETUNED_CKPT in legacy single mode. Otherwise use CHECKPOINT_DIR,
# defaulting to the newest run's checkpoints directory.
if [ -n "${FINETUNED_CKPT:-}" ]; then
    CHECKPOINT_ARG=(--finetuned-ckpt "$FINETUNED_CKPT")
else
    if [ -z "${CHECKPOINT_DIR:-}" ]; then
        CHECKPOINT_DIR="$(ls -td "$POC"/outputs/finetune_*/checkpoints 2>/dev/null | head -n1 || true)"
    fi
    CHECKPOINT_ARG=(--checkpoint-dir "$CHECKPOINT_DIR")
fi
if [ -z "${FINETUNED_CKPT:-}" ] && [ -z "${CHECKPOINT_DIR:-}" ]; then
    echo "ERROR: no checkpoint directory found under $POC/outputs/finetune_*/checkpoints/" >&2
    echo "Run finetune.slurm first, or set CHECKPOINT_DIR or FINETUNED_CKPT explicitly." >&2
    exit 1
fi
OUT_DIR="$POC/outputs/evaluate_${SLURM_JOB_ID:-local}"

mkdir -p logs "$OUT_DIR"
export PYTHONPATH="$GENMOL_ROOT/src:${PYTHONPATH:-}"

python "$POC/scripts/generate_evaluate.py" \
    --base-ckpt "$PRETRAINED_CKPT" \
    "${CHECKPOINT_ARG[@]}" \
    --train-smiles "$DATA_DIR/train_smiles.txt" \
    --ref-smiles "$DATA_DIR/val_smiles.txt" \
    --length-source "${LENGTH_SOURCE:-train}" \
    --out-dir "$OUT_DIR" \
    --num-samples 1000 \
    $V2_FLAG

echo "Evaluation finished. Results: $OUT_DIR"
