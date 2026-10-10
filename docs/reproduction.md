# Reproduction guide

## 1. Scope and status

This pre-release repository publishes code, not the curated dataset or its
construction pipeline. Only **Geometric Transformer (Qin-style)** is trainable
here. External-baseline evaluators consume already-generated predictions.
Synthetic tests run now; reproducing manuscript numbers requires an authorized
manifest, matched experimental coordinates, predictions/checkpoints, and recorded
run configurations.

Execute the commands from the repository root unless stated otherwise. Use a
compatible PyTorch installation and `pip install -r requirements.txt`.
The supplied shell scripts run locally without a scheduler.

## 2. Set data paths

```bash
export MS_ROOT="$(pwd)"
export MUTASTRUCT_MANIFEST=/path/to/tier_a_pairs_with_splits.parquet
export MUTASTRUCT_STRUCTURES=/path/to/structures
export MUTASTRUCT_CACHE=/path/to/private/graph_v1_r6_n32
```

The manifest may be CSV or Parquet. The structure loader uses polymer-entity and
`label_asym_id` chain identifiers, not assumed author-chain or PDB-file row order.
Read [dataset_schema.md](dataset_schema.md) before preparing an input.

Model tensor caches contain experimental target coordinates and dataset
identifiers; they are generated locally and are not distributed here.
Reuse an existing compatible cache only if the manifest, sequence mapping,
coordinate inputs, cache schema, graph radius, and neighbor limit agree. The
legacy loader checks a schema version, not a complete source-data fingerprint.

## 3. Model-input cache preparation (requires data)

```bash
python geometric_transformer/prepare_cache.py \
  --manifest "$MUTASTRUCT_MANIFEST" \
  --structures-dir "$MUTASTRUCT_STRUCTURES" \
  --cache-dir "$MUTASTRUCT_CACHE" \
  --splits train validation test --workers 8
```

This reads an existing locked dataset. It is not the withheld pair-curation or
split-generation pipeline. Resolve reported errors before using partial caches.

## 4. Optional training

```bash
CUDA_VISIBLE_DEVICES=0 python geometric_transformer/train.py \
  --manifest "$MUTASTRUCT_MANIFEST" \
  --structures-dir "$MUTASTRUCT_STRUCTURES" \
  --cache-dir "$MUTASTRUCT_CACHE" \
  --run-dir "$MS_ROOT/runs/geometric_transformer" \
  --epochs 100 --batch-size 2 --gradient-accumulation 4 \
  --learning-rate 1e-4 --weight-decay 1e-4 --patience 15 \
  --workers 8 --seed 42 --monitor loss_total --amp
```

The starting objective is coordinate loss + 0.20 edge-distance loss + 0.02 clash
loss. Experimental response-aware terms remain zero in the default configuration.
The supplied `configs/geometric_transformer_defaults.json` records **example
architecture/training defaults**, not the settings of every trained checkpoint.
When provided, `configs/released_run.json` records the available saved settings
associated with the distributed checkpoint.

Use `--max-train 64 --max-validation 32 --epochs 2` and a separate `--run-dir` for a
real-data smoke run. For CPU, set `--device cpu --no-amp`. The model uses a 12 Å crop and the standard coordinate, edge-distance, and
clash objective; it does not model arbitrary whole-protein rearrangements.

The same CLI has `--resume` for trusted full training checkpoints and
`--init-checkpoint` for weights-only initialization. Public sanitized checkpoints
omit optimizer/scheduler state and are for inference or fresh-optimizer
initialization, **not exact training resume**.

## 5. Geometric Transformer evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python geometric_transformer/evaluate.py \
  --checkpoint checkpoints/geometric_transformer_best.pt \
  --manifest "$MUTASTRUCT_MANIFEST" \
  --structures-dir "$MUTASTRUCT_STRUCTURES" \
  --cache-dir "$MUTASTRUCT_CACHE" \
  --split test --output-dir results/geometric_transformer --amp
```

Use `--split validation` for development, and `--max-cases 32` for a smoke run.
The evaluator runs the local crop and copies the unchanged initialization outside
it before evaluating the full represented chain. Its model label is exactly
`Geometric Transformer` and denotes **only the Qin-style architecture**.

Public evaluation uses restricted checkpoint loading. The supported format
contains `model_state_dict`, `configuration`, and `epoch`. Optimizer and scheduler
state are not required for inference. Do not load an untrusted checkpoint using
unrestricted pickle loading.

## 6. Controls

```bash
python evaluation/evaluate_controls.py \
  --manifest "$MUTASTRUCT_MANIFEST" \
  --structures-dir "$MUTASTRUCT_STRUCTURES" \
  --cache-dir "$MUTASTRUCT_CACHE" \
  --output-dir results/controls
```

This scores Copy-WT and Mutant-Init on the test split. Copy-WT's primary lDDT and
side-chain columns remain unavailable by protocol; separate common-atom
diagnostics must not be substituted into the full-mutant columns.

## 7. Existing external predictions

These commands **evaluate predictions; they do not run the external models**.
The external scripts currently select the manifest's test split.

### AlphaFold 3

```bash
python evaluation/evaluate_alphafold3.py \
  --manifest "$MUTASTRUCT_MANIFEST" \
  --structures-dir "$MUTASTRUCT_STRUCTURES" \
  --cache-dir "$MUTASTRUCT_CACHE" \
  --af3-structures /path/to/af3_tier_a_target_structures \
  --af3-mapping /path/to/test_pairs_af3_targets.csv \
  --output-dir results/alphafold3
```

Expected prediction filenames are `af3_<sequence_sha256_prefix>_model.cif`.
The external CSV maps `pair_id` to `target_prediction_id`. It is not included in
this repository. Use the mapping for the exact AF3 input set: do not substitute
old mappings or assume manifest-sequence hashes always match the AF3 polymer
input hashes. The evaluator verifies the predicted standard-amino-acid chain
against the filename hash. It is not a universal modified-residue AF3 parser.

### ESMFold, PreMut, FoldX, Rosetta

```bash
python evaluation/evaluate_baselines.py \
  --manifest "$MUTASTRUCT_MANIFEST" \
  --structures-dir "$MUTASTRUCT_STRUCTURES" \
  --cache-dir "$MUTASTRUCT_CACHE" \
  --models esmfold premut foldx rosetta \
  --esmfold-root /path/to/ESMFold/results/facebook_esmfold_v1/predictions \
  --premut-root /path/to/PreMut/mutastruct_test_predictions \
  --foldx-root /path/to/FoldX/results/predictions \
  --rosetta-root /path/to/Rosetta/results/predictions \
  --output-dir results/external_baselines
```

Expected legacy layouts: ESMFold `SEQ_<full-sha256>.pdb` (recursive search);
PreMut `pair_to_prediction.csv` or `pair_predictions/<pair_id>.pdb`;
FoldX/Rosetta input-key job folders containing `predicted_mutant_A.pdb`.
Use `--foldx-map-csv` / `--rosetta-map-csv` for explicit mappings where needed;
the mapper accepts `pair_id` plus a prediction-path column. Ambiguous nonidentical
files are rejected under the default strict mapping behavior.

Select a subset of methods with `--models`. Third-party software, licenses, model
weights, seeds and prediction configurations must be obtained/recorded separately.
The optional [PreMut adapter](../baselines/README.md) is not a PreMut distribution.

## 8. Larger-response post-hoc evaluation (requires private labels)

The code to *filter existing results* is public; the list of events is not.

```bash
python evaluation/build_hard_subset.py \
  --manifest "$MUTASTRUCT_MANIFEST" \
  --copywt-pair-metrics results/controls/pair_metrics.csv \
  --threshold 1.0 --expected-events 33 --expected-pairs 75 \
  --output-dir results/hard_subset_local_ca_gt1

python evaluation/evaluate_hard_subset.py \
  --hard-subset-dir results/hard_subset_local_ca_gt1 \
  --result results/controls/pair_metrics.csv \
  --result results/geometric_transformer/pair_metrics.csv \
  --result results/alphafold3/pair_metrics.csv \
  --result results/external_baselines/pair_metrics.csv \
  --no-discover --output-dir results/hard_subset_evaluation \
  --bootstrap 10000 --seed 42
```

Selection uses the event **median** native local Cα change >1.0 Å; retain all
coordinate pairs of each selected event. Scoring still averages each model's
pair metrics by event **mean**, then equally across events. The expected 33/75
counts are checks for the locked dataset, not universal properties of a new input.
Prefer explicit source files (`--no-discover`) to avoid accidentally choosing
among different checkpoints/runs with the same model label. No predictions or
atomic metrics are recomputed by this filtering step.

## 9. What to check before reporting numbers

- Record code revision, checkpoint SHA-256, model/data configuration, numerical
  precision, input mapping provenance and dataset version.
- Examine pair/event coverage, failed cases, and metric-specific nonmissing counts.
- Use `event_metrics.csv` / event-macro summaries, not pair means, for the primary
  benchmark. Different scripts retain their historical summary-column names.
- Do not turn missing side-chain scores into zero. Read the metric protocol.
- The native-response subset is target-defined and was explored after observing
  the benchmark; do not use its test membership for further model selection.
- Internal train/validation separation cannot by itself rule out pretrained-model
  exposure to overlapping public protein structures.

Numerical reproduction depends on consistent metric implementations,
checkpoint settings, and prediction-to-target mappings.
