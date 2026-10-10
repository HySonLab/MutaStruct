#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/geometric_transformer/evaluate.py" \
  --checkpoint "$ROOT/checkpoints/geometric_transformer_best.pt" \
  --output-dir "$ROOT/results/geometric_transformer" \
  --split test --amp "$@"
