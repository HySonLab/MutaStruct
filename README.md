# MutaStruct

Model training and evaluation code for benchmarking experimentally observed
protein structural changes associated with single-site amino-acid substitutions.

## Scope and availability

MutaStruct distinguishes mutation events from their repeated experimental
coordinate observations. Evaluation uses complementary global, mutation-local,
and reference-relative structural metrics, with event-macro aggregation.

This repository contains:

- **Geometric Transformer**: a Qin-style distance/triplet-angle attention model
  with an equivariant coordinate-refinement decoder, plus its training interface.
- Evaluation of existing AlphaFold 3, ESMFold, PreMut, FoldX, and Rosetta predictions.
- Copy-WT and Mutant-Init diagnostic controls.
- Filtering and reaggregation of existing results for the larger-response subset.
- Dataset-schema documentation, metric definitions, and synthetic tests.
- An optional PreMut inference adapter; the upstream implementation is separate.

The codebase, trained models, evaluation pipeline, and dataset specifications are
publicly available. The curated benchmark dataset and dataset-construction
pipeline are not included in the current release because they are undergoing
additional quality control, validation, and expansion for the journal version of
the benchmark. These resources will be released in a future update.

The only GT architecture in this repository is the **Geometric Transformer**
described in docs/geometric_transformer.md.
Third-party model distributions, weights, and databases are not bundled.

## Installation and tests

Use Python 3.11 or later and a PyTorch installation suitable for your hardware:

```bash
conda create -n mutastruct python=3.11 -y
conda activate mutastruct
python -m pip install torch
python -m pip install -r requirements.txt
bash scripts/run_tests.sh
```

The tests use artificial structures and manifests in temporary directories.
They do not require the MutaStruct dataset, pretrained weights, or a GPU.
GPU training requires an appropriate CUDA-enabled PyTorch installation.

## Repository layout

```text
MutaStruct/
├── geometric_transformer/   # model, input loader, training and evaluation
├── evaluation/              # external predictions, controls, subset evaluation
├── baselines/premut/        # optional inference adapter
├── configs/                 # default and checkpoint-specific configuration
├── checkpoints/             # model weights, when distributed
├── docs/                    # dataset schema, method and reproduction details
├── scripts/                 # local training, evaluation and test commands
├── tests/
└── tools/                   # optional checkpoint export utility
```

`geometric_transformer/prepare_cache.py` converts an existing manifest and
structures into model tensors. It is not the dataset-curation or split-generation
pipeline.

## Training and evaluation

With access to the required data, set the paths explicitly:

```bash
export MUTASTRUCT_MANIFEST=/path/to/tier_a_pairs_with_splits.parquet
export MUTASTRUCT_STRUCTURES=/path/to/structures
export MUTASTRUCT_CACHE=/path/to/model_cache

CUDA_VISIBLE_DEVICES=0 bash scripts/train_geometric_transformer.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_geometric_transformer.sh
```

The evaluation script expects `checkpoints/geometric_transformer_best.pt` by
default. When a checkpoint is distributed, its hash and available run settings
are recorded in `checkpoints/release_metadata.json` and
`configs/released_run.json`. The default configuration is not a substitute for
those run-specific settings.

See [reproduction.md](docs/reproduction.md) for command-line options, external
prediction layouts, required data, and coverage checks.

## Documentation

- [Dataset schema](docs/dataset_schema.md)
- [Reproduction guide](docs/reproduction.md)
- [Evaluation protocol](docs/evaluation_protocol.md)
- [Geometric Transformer architecture](docs/geometric_transformer.md)
- [Third-party baselines](baselines/README.md)

## Attribution and licensing

The Geometric Transformer adapts Qin et al.'s geometric attention ideas for
mutation-conditioned coordinate refinement, rather than reproducing their
original point-cloud registration model.

See [NOTICE.md](NOTICE.md) for method attribution and
[LICENSING.md](LICENSING.md) for the current licensing status.

## Citation

```bibtex
@article {Huynh2026.10.21.760837,
	author = {Huynh, Tan-Phat and Ho, Dang-Manh and Nguyen, Viet Thanh Duy and Kha, Quang-Hien and Pham, Ngoc-Thac and Lai, Kuei-Hung and Hy, Truong-Son},
	title = {MutaStruct: Revisiting Benchmarks for Single-Site Mutation-Induced Structural Change Prediction},
	elocation-id = {2026.10.21.760837},
	year = {2026},
	doi = {10.64898/2026.10.21.760837},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Predicting the structural consequences of single-site mutations is fundamental to understanding protein function, disease mechanisms, and protein engineering. However, existing benchmarks for mutation-induced structural change prediction are often affected by duplicate mutation events, highly similar structural pairs, and experimental-state confounders, which can inflate reported performance. We introduce MutaStruct, a curated benchmark of experimentally observed single-mutation protein structure pairs designed for rigorous evaluation of mutation-induced structural changes. MutaStruct combines residue-level sequence validation, coordinate-level quality control, context-aware filtering, event-level deduplication, and leakage-aware data splitting to improve dataset reliability and evaluation rigor. We benchmark sequence-only structure prediction, physics-based refinement, and learning-based refinement methods together with a Copy-WT diagnostic control. Copy-WT remains highly competitive on backbone-based metrics, while method rankings change substantially across local and mutation-site measures. These results show that overall structural fidelity does not necessarily imply accurate modeling of mutation-associated structural change, underscoring the need for further methodological development beyond current paradigms. The benchmark dataset is currently undergoing further curation and expansion and will be released in a future journal-version update. The codebase, trained models, and evaluation pipeline are publicly available at https://github.com/HySonLab/MutaStructCompeting Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2026/10/22/2026.10.21.760837},
	eprint = {https://www.biorxiv.org/content/early/2026/10/22/2026.10.21.760837.full.pdf},
	journal = {bioRxiv}
}
```
