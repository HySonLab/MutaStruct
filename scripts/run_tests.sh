#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
cd "$ROOT/geometric_transformer"
for test in smoke_test geometric_embedding_test end_to_end_smoke_test trainer_smoke_test evaluator_smoke_test; do
  python "tests/${test}.py"
done
cd "$ROOT"
python tests/test_checkpoint_export.py
python tests/test_evaluation_adapters.py
