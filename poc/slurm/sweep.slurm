#!/bin/bash
#SBATCH --job-name=genmol-sweep
#SBATCH --partition=req-576              # <-- adjust to your cluster's A100 partition
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-8                  # 3 freeze x 3 lr = 9 runs (one A100 each)
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

# GenMol PoC — hyperparameter sweep orchestrator (freeze-layers x lr grid).
# Each array task is ONE run: it calls the single-run scripts unchanged
# (finetune.py -> generate_evaluate.py) for its (freeze, lr) combination.
# After the array finishes, rank + select with:
#   python poc/scripts/aggregate_sweep.py poc/outputs/sweep
#
# Submit from the workspace root (logs/ must exist) AFTER the data split is prepared
# (run prepare_data.py or finetune.slurm once first). Override the grid via env:
#   sbatch --export=ALL,FREEZE="0 6 9",LRS="5e-5 1e-4 2e-4" poc/slurm/sweep.slurm
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate genmol

WORKSPACE="${WORKSPACE:-$SLURM_SUBMIT_DIR}"
GENMOL_ROOT="$WORKSPACE/genmol"
POC="$WORKSPACE/poc"
DATA_DIR="$POC/outputs/data"
export PYTHONPATH="$GENMOL_ROOT/src:${PYTHONPATH:-}"

# V2 by default (model_v2.ckpt + bracket SAFE). For V1, set MODEL_VERSION=v1.
MODEL_VERSION="${MODEL_VERSION:-v2}"
if [ "$MODEL_VERSION" = "v1" ]; then
    PRETRAINED_CKPT="${PRETRAINED_CKPT:-$WORKSPACE/models/genmol_v1_v1.0/model.ckpt}"
    BRACKET_FLAG="--no-use-bracket-safe"; V2_FLAG="--no-v2"
else
    PRETRAINED_CKPT="${PRETRAINED_CKPT:-$WORKSPACE/models/genmol_v2_v1.0/model_v2.ckpt}"
    BRACKET_FLAG="--use-bracket-safe"; V2_FLAG="--v2"
fi
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: pretrained checkpoint not found: $PRETRAINED_CKPT" >&2
    exit 1
fi

# --- Grid: freeze-layers x lr (index maps array task -> combo) ---
read -r -a FREEZE <<< "${FREEZE:-0 6 9}"
read -r -a LRS <<< "${LRS:-5e-5 1e-4 2e-4}"
N_LR=${#LRS[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}
freeze=${FREEZE[$(( IDX / N_LR ))]}
lr=${LRS[$(( IDX % N_LR ))]}
TAG="f${freeze}_lr${lr}"
OUT="$POC/outputs/sweep/$TAG"
NUM_SAMPLES="${NUM_SAMPLES:-300}"     # smaller than the final run — this is a screen
mkdir -p "$OUT"
echo "[$TAG] freeze=$freeze lr=$lr num_samples=$NUM_SAMPLES -> $OUT"

# --- 1) fine-tune this combo (single-run script, unchanged) ---
python "$POC/scripts/finetune.py" \
    --pretrained-ckpt "$PRETRAINED_CKPT" \
    --data "$DATA_DIR/train.safe" \
    --out-dir "$OUT" \
    --freeze-layers "$freeze" --lr "$lr" \
    $BRACKET_FLAG 2>&1 | tee "$OUT/finetune.log"

# capture best val_loss for ranking (printed by finetune.py)
grep -Eo "Best val_loss: [0-9.]+" "$OUT/finetune.log" | tail -n1 | awk '{print $3}' > "$OUT/val_loss.txt" || true

# --- 2) evaluate this combo's best checkpoint (single-run script, unchanged) ---
CKPT="$OUT/checkpoints/best.ckpt"
[ -f "$CKPT" ] || CKPT="$(ls -t "$OUT"/checkpoints/*.ckpt 2>/dev/null | head -n1 || true)"
if [ -z "$CKPT" ]; then
    echo "[$TAG] ERROR: no checkpoint produced" >&2; exit 1
fi
python "$POC/scripts/generate_evaluate.py" \
    --base-ckpt "$PRETRAINED_CKPT" \
    --finetuned-ckpt "$CKPT" \
    --train-smiles "$DATA_DIR/train_smiles.txt" \
    --ref-smiles "$DATA_DIR/val_smiles.txt" \
    --out-dir "$OUT/eval" \
    --num-samples "$NUM_SAMPLES" \
    $V2_FLAG

echo "[$TAG] done -> $OUT/eval/comparison.json"
