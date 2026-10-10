#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Choose GPU externally, e.g. CUDA_VISIBLE_DEVICES=0 bash scripts/train_geometric_transformer.sh
python "$ROOT/geometric_transformer/train.py" \
  --run-dir "$ROOT/runs/geometric_transformer" \
  --epochs 100 --batch-size 2 --gradient-accumulation 4 \
  --learning-rate 1e-4 --weight-decay 1e-4 --workers 8 \
  --seed 42 --monitor loss_total --amp "$@"
