#!/bin/bash
#SBATCH --job-name=genmol-finetune
#SBATCH --partition=gpu              # <-- adjust to your cluster's A100 partition
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

# GenMol PoC — data prep + fine-tuning on one A100. Edit paths/modules for your cluster.
# NOTE: submit from the workspace root so the logs/ dir in --output/--error exists.
set -euo pipefail

# --- Environment -----------------------------------------------------------
# module load cuda/12.1                      # if your cluster uses modules
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate genmol

# --- Paths -----------------------------------------------------------------
WORKSPACE="${WORKSPACE:-$SLURM_SUBMIT_DIR}"           # workspace root (genmol-poc/)
GENMOL_ROOT="$WORKSPACE/genmol"
POC="$WORKSPACE/poc"
RAW_DATA="$WORKSPACE/data/delaney-processed.csv"
# V2 by default (uses model_v2.ckpt + bracket SAFE). For V1, set MODEL_VERSION=v1.
MODEL_VERSION="${MODEL_VERSION:-v2}"
if [ "$MODEL_VERSION" = "v1" ]; then
    PRETRAINED_CKPT="${PRETRAINED_CKPT:-$WORKSPACE/models/genmol_v1_v1.0/model.ckpt}"
    BRACKET_FLAG="--no-use-bracket-safe"
else
    PRETRAINED_CKPT="${PRETRAINED_CKPT:-$WORKSPACE/models/genmol_v2_v1.0/model_v2.ckpt}"
    BRACKET_FLAG="--use-bracket-safe"
fi
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: pretrained checkpoint not found: $PRETRAINED_CKPT" >&2
    exit 1
fi
OUT_DIR="$POC/outputs/finetune_${SLURM_JOB_ID:-local}"
DATA_DIR="$POC/outputs/data"

mkdir -p logs "$OUT_DIR" "$DATA_DIR"
export PYTHONPATH="$GENMOL_ROOT/src:${PYTHONPATH:-}"

# --- 1) Data preparation (SMILES -> SAFE, split, stats) --------------------
python "$POC/scripts/prepare_data.py" \
    --input "$RAW_DATA" \
    --out-dir "$DATA_DIR"

# --- 2) Fine-tune the pretrained GenMol model ------------------------------
python "$POC/scripts/finetune.py" \
    --pretrained-ckpt "$PRETRAINED_CKPT" \
    --data "$DATA_DIR/train.safe" \
    --out-dir "$OUT_DIR" \
    $BRACKET_FLAG

echo "Fine-tuning finished. Checkpoints: $OUT_DIR/checkpoints"
