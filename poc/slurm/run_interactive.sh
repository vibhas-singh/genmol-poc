#!/bin/bash
# Run the full GenMol PoC pipeline in an interactive Slurm session (salloc / srun --pty).
# Assumes you already have a GPU allocation and the conda env active:
#   salloc --partition=gpu --gres=gpu:a100:1 --cpus-per-task=16 --mem=64G --time=02:00:00
#   conda activate genmol
#
# Usage (from the workspace root, genmol-poc/):
#   bash poc/slurm/run_interactive.sh
#
# Env overrides:
#   MODEL_VERSION=v1|v2   (default v2)          PRETRAINED_CKPT=/path/to/ckpt
#   NUM_SAMPLES=1000      (eval sample count)   MAX_EPOCHS=<n>  (fine-tune epoch cap)
#   SMOKE=1               (quick smoke test: MAX_EPOCHS=2, NUM_SAMPLES=100, no early stop)
set -euo pipefail

WORKSPACE="${WORKSPACE:-$PWD}"
GENMOL_ROOT="$WORKSPACE/genmol"
POC="$WORKSPACE/poc"
RAW_DATA="$WORKSPACE/data/delaney-processed.csv"
DATA_DIR="$POC/outputs/data"
OUT_DIR="$POC/outputs/interactive"
export PYTHONPATH="$GENMOL_ROOT/src:${PYTHONPATH:-}"

# V2 by default (model_v2.ckpt + bracket SAFE); set MODEL_VERSION=v1 for the V1 model.
MODEL_VERSION="${MODEL_VERSION:-v2}"
if [ "$MODEL_VERSION" = "v1" ]; then
    PRETRAINED_CKPT="${PRETRAINED_CKPT:-$WORKSPACE/models/genmol_v1_v1.0/model.ckpt}"
    BRACKET_FLAG="--no-use-bracket-safe"
    V2_FLAG="--no-v2"
else
    PRETRAINED_CKPT="${PRETRAINED_CKPT:-$WORKSPACE/models/genmol_v2_v1.0/model_v2.ckpt}"
    BRACKET_FLAG="--use-bracket-safe"
    V2_FLAG="--v2"
fi
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: pretrained checkpoint not found: $PRETRAINED_CKPT" >&2
    exit 1
fi

# Smoke test shortcut: tiny run to validate the whole path in minutes.
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
FT_EXTRA=""
if [ "${SMOKE:-0}" = "1" ]; then
    FT_EXTRA="--max-epochs 2 --patience 0"
    NUM_SAMPLES=100
elif [ -n "${MAX_EPOCHS:-}" ]; then
    FT_EXTRA="--max-epochs $MAX_EPOCHS"
fi

mkdir -p "$DATA_DIR" "$OUT_DIR"

echo "== GPU check =="
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA on this node'; print('GPU:', torch.cuda.get_device_name(0))"

echo "== 1/3 data preparation =="
python "$POC/scripts/prepare_data.py" --input "$RAW_DATA" --out-dir "$DATA_DIR"

echo "== 2/3 fine-tuning ($MODEL_VERSION) =="
python "$POC/scripts/finetune.py" \
    --pretrained-ckpt "$PRETRAINED_CKPT" \
    --data "$DATA_DIR/train.safe" \
    --out-dir "$OUT_DIR" \
    $BRACKET_FLAG $FT_EXTRA

# Prefer the best-on-val checkpoint, else the newest. `|| true` keeps a no-match `ls`
# from tripping `set -o pipefail` (e.g. SMOKE runs that never write best.ckpt).
FINETUNED_CKPT="$(ls -t "$OUT_DIR"/checkpoints/best.ckpt 2>/dev/null | head -n1 || true)"
FINETUNED_CKPT="${FINETUNED_CKPT:-$(ls -t "$OUT_DIR"/checkpoints/*.ckpt 2>/dev/null | head -n1 || true)}"
if [ -z "$FINETUNED_CKPT" ]; then
    echo "ERROR: no checkpoint produced under $OUT_DIR/checkpoints/" >&2
    exit 1
fi

echo "== 3/3 generation + evaluation =="
python "$POC/scripts/generate_evaluate.py" \
    --base-ckpt "$PRETRAINED_CKPT" \
    --finetuned-ckpt "$FINETUNED_CKPT" \
    --train-smiles "$DATA_DIR/train_smiles.txt" \
    --ref-smiles "$DATA_DIR/val_smiles.txt" \
    --out-dir "$OUT_DIR" \
    --num-samples "$NUM_SAMPLES" \
    $V2_FLAG

echo "Done. Outputs in: $OUT_DIR"
