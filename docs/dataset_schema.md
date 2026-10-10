# MutaStruct dataset schema

**Status:** schema documentation only; the curated data are not included in this repository before the dataset release.

This document specifies the existing Tier-A reference–mutant **pair manifest** used by the MutaStruct training and evaluation code. It documents how to interpret future dataset files; it does **not** describe or distribute the private dataset-construction pipeline, actual pair mappings, mutation labels, train/validation/test membership records, or raw processed caches.

## 1. Data units and identifiers

| Unit | Identifier | Interpretation |
|---|---|---|
| Coordinate pair | `pair_id` | One selected experimental reference structure/chain paired with one experimental mutant structure/chain. Unique per manifest row. |
| Mutation event | `event_id` | One sequence-background/mapped-interval substitution event. One event may have multiple coordinate-pair observations. |
| Split cluster | `split_cluster_id` | Indivisible leakage-control group used to partition related sequences/events into a single split. It is **not** a structure-similarity cluster. |

**Do not treat coordinate pairs from the same `event_id` as independent mutations.** Primary benchmark scores first average valid coordinate-pair metrics within an event, then average across events with equal weight.

## 2. Pair-manifest fields

The current manifest has **20 columns**. A future release may distribute the same information as `.csv` or `.parquet` (the GT loader accepts both). Columns marked *core* are needed by the current GT data loader; *benchmark* fields support grouping and reporting; *legacy* fields are retained for compatibility but should not be used in place of the unambiguous PDB/chain fields.

| Column | Type | Role | Meaning |
|---|---|---|---|
| `pair_id` | string | core | Stable within a frozen dataset release; unique coordinate-pair identifier. |
| `event_id` | string | benchmark | Identifier grouping structural observations of the same substitution event. |
| `split_cluster_id` | string | benchmark | Sequence-linked grouping unit used for the locked split. |
| `split` | categorical string | core | `train`, `validation`, or `test`. |
| `pair_class` | categorical string | benchmark | See the three pair classes below. |
| `reference_pdb_id` | string | core | PDB entry containing the reference chain. |
| `reference_entity_id` | integer/string | core | Reference polymer entity identifier in the mmCIF entry. |
| `reference_chain_label_asym_id` | string | core | Reference mmCIF **label** asymmetric-unit chain ID (not necessarily author chain ID). |
| `reference_sequence` | string | core | One-letter reference polymer sequence for the compared construct. |
| `reference_aa` | one-letter string | benchmark | Reference amino acid at the substitution site. |
| `target_pdb_id` | string | core | PDB entry containing the experimentally observed mutant chain. |
| `target_entity_id` | integer/string | core | Mutant polymer entity identifier in the mmCIF entry. |
| `target_chain_label_asym_id` | string | core | Mutant mmCIF **label** asymmetric-unit chain ID. |
| `target_sequence` | string | core | One-letter mutant polymer sequence, not the model prediction. |
| `target_aa` | one-letter string | benchmark | Mutant amino acid at the substitution site. |
| `mutation_index_0` | integer | core | **Zero-based** substitution index into both manifest sequences. |
| `mutation` | string | benchmark | Human-readable one-letter substitution, e.g. `A42V` (1-based position). |
| `mutation_info` | string | legacy | Internal mutation description; prefer `mutation_index_0` and amino-acid columns. |
| `wild` | string | legacy | Internal reference-chain alias; do not interpret as a canonical-wild-type guarantee. |
| `mutant` | string | legacy | Internal mutant-chain alias; prefer explicit PDB, entity, and chain columns. |

### Pair-class values

| Value | Interpretation |
|---|---|
| `canonical_wt` | Reference construct matches the canonical background over the mapped interval. |
| `background_matched` | A valid single substitution between matched constructs, but the reference is not necessarily canonical wild type. |
| `mapped_core_length_mismatch` | Full compared constructs still represent a valid equal-length single substitution, but their *resolved or mapped structural coverage* differs. This is **not** permission to include insertions or deletions. |

### Key invariants in the locked Tier-A split

- Exactly one sequence substitution: `len(reference_sequence) == len(target_sequence)`, with one differing character at `mutation_index_0`.
- `reference_sequence[mutation_index_0] == reference_aa` and `target_sequence[mutation_index_0] == target_aa`.
- `mutation_index_0` is zero-based; display strings such as `A42V` are one-based.
- Each `pair_id` identifies exactly one row and belongs to one split.
- Each `event_id` and each `split_cluster_id` belongs entirely to one split.
- The locked split was also audited for shared exact reference/variant sequences, UniProt accessions, and PDB polymer entities across splits. These are *release-level invariants*, not guarantees that every field is present in this minimal manifest.
- An event may have multiple experimental reference–mutant coordinate pairs. Resolved atoms or local context can differ across observations.

## 3. Coordinate sources and model-ready representation

The pair manifest **identifies** the reference and experimental mutant structures; it does not embed atomic coordinates. The current GT loader uses the specified polymer entities and `label_asym_id` chains from the respective mmCIF entries, for example from a local structure directory containing `<PDB_ID>.cif.gz` files.

In a model-ready example, inputs are constructed from the **reference coordinates plus mutation identity**. Mutant-only side-chain atoms are initialized without consulting experimental mutant coordinates. The experimental mutant coordinates are used as masked supervision/evaluation targets, **never as prediction inputs**.

The model cache (`*.pt`) is an internal derivative used for speed, **not** the canonical released dataset format. Representative tensor keys include `pos_init`, `pos_target`, `target_mask`, `atom_type`, `res_type_wt`, `res_type_mt`, `is_mutation_site`, `is_new_atom`, `is_ca`, and graph edges. Cache schema and atomic masks may change with code versions; use the supported loader rather than relying on an undocumented tensor order.

## 4. Split sizes (Tier A, locked version described in the manuscript)

| Split | Coordinate pairs | Mutation events |
|---|---:|---:|
| Train | 10,635 | 6,522 |
| Validation | 1,527 | 798 |
| Test | 1,276 | 827 |
| **Total** | **13,438** | **8,147** |

These counts describe the locked benchmark version, not a promise that later releases will have identical contents. Cite the dataset version when using the results.

## 5. Evaluation output schema (not part of the input manifest)

The model evaluators write these results in their selected output directory:

| File | One row per | Relevant fields |
|---|---|---|
| `pair_metrics.csv` | model × coordinate pair | `model`, `pair_id`, `event_id`, five metrics, available atom/residue counts |
| `event_metrics.csv` | model × mutation event | `model`, `event_id`, event-averaged five metrics |
| `main_table_event_macro.csv` | model | Event-macro mean for each of the five metrics |
| `summary_pair_macro.csv` | model and/or metric | Pair-level summary (secondary) |
| `summary_event_macro.csv` | model and/or metric | Event-level summary (primary) |
| `evaluation_metadata.json` | evaluation run | Configuration/provenance of the run |

The five public metric column names are:

```text
ca_rmsd
lddt
local_ca_rmsd_8a
mutation_sidechain_rmsd
mutation_response_rmsd_8a
```

Their meanings are defined in [`evaluation_protocol.md`](evaluation_protocol.md). Missing or inapplicable atoms are handled with evaluation masks. Mutation-site side-chain RMSD may be unavailable for particular substitutions and baseline types; do not replace missing scores by zero, and report the effective event count.

## 6. Planned challenge subset

After data release, a larger-response challenge subset can be *derived* using the **event-level median** of the experimentally observed local Cα RMSD in the reference-defined 8 Å neighborhood. A strict threshold of **greater than 1.0 Å** produced **33 of the 827 test mutation events** (75 coordinate-pair observations) in the locked analysis. All pairs from qualifying events are retained. This is a **target-defined evaluation stratum**; its membership must not be provided to a prediction model at inference or used to tune that model on test.

## 7. Pre-release availability

Only this **schema** is public at present. The curated manifest, actual identifiers/rows, full response labels, and challenge-subset membership are intentionally excluded. Code-level smoke tests are available without the dataset. Full benchmark reproduction requires the versioned data package or separately authorized reviewer access.
