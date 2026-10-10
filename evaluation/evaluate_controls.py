#!/usr/bin/env python3
"""Evaluate two no-learning controls on the locked MutaStruct Tier-A test set.

Controls
--------
1) Copy-WT
   The predicted variant backbone is simply the experimental WT/reference
   structure, unchanged.

   This is the critical "does the model beat doing nothing?" baseline.

   Three primary metrics are directly comparable:
     - Global Cα RMSD ↓
     - Local Cα RMSD (8 Å) ↓
     - MR-RMSD (8 Å) ↓

   A literal WT structure does NOT contain mutant-only side-chain atoms.
   Therefore the main-table values for:
     - all-atom lDDT
     - mutation-site side-chain RMSD
   are deliberately left as NA for Copy-WT rather than inventing coordinates.

   For diagnostic purposes only, the script additionally calculates
   common-atom lDDT and common-side-chain RMSD using atoms present in both the
   WT-derived graph and experimental target. These diagnostics are saved
   separately and should NOT be placed in the same table columns as the full
   all-atom model metrics.

2) Mutant-Init
   The exact chemically plausible mutant initialization supplied to benchmark,
   with NO learned coordinate refinement.

   This control is fully evaluable with all five main metrics and answers:
       "Does the neural network improve over its own preprocessing/input?"

Recommended manuscript interpretation
-------------------------------------
A learned refinement model should ideally improve over:
  Copy-WT      on Cα RMSD, Local Cα RMSD8Å, and MR-RMSD8Å
and
  Mutant-Init  on all five metrics.

Run:
    cd geometric_transformer
    python evaluate_trivial_baselines.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1] / "geometric_transformer"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.constants import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_STRUCTURES_DIR,
)
from common.data import PairGraphDataset  # noqa: E402
from common.graph import GraphBuildConfig  # noqa: E402


DEFAULT_OUTPUT = ROOT.parent / "results" / "controls"

BACKBONE_METRICS = [
    "ca_rmsd",
    "local_ca_rmsd_8a",
    "mutation_response_rmsd_8a",
]

ALL_FIVE = [
    "ca_rmsd",
    "lddt",
    "local_ca_rmsd_8a",
    "mutation_sidechain_rmsd",
    "mutation_response_rmsd_8a",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--structures-dir", type=Path, default=DEFAULT_STRUCTURES_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--pair-class", default="all")
    p.add_argument("--local-radius", type=float, default=8.0)
    p.add_argument("--lddt-radius", type=float, default=15.0)
    return p.parse_args()


def load_main_evaluator():
    """Import metric definitions from the evaluator already used for benchmark.

    Uses the bundled Geometric Transformer evaluator (no excluded models).
    """
    candidates = [
        ROOT / "evaluate.py",
    ]

    for path in candidates:
        if not path.exists():
            continue

        spec = importlib.util.spec_from_file_location(
            "_mutastruct_main5_evaluator",
            path,
        )
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        required = [
            "compute_metrics",
            "globally_align_prediction_to_target",
            "symmetry_correct_prediction",
            "atom_names_for_sample",
            "BACKBONE_NAMES",
        ]
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise RuntimeError(
                f"{path} exists but is missing evaluator functions: {missing}"
            )

        print(f"Using shared metric implementation from: {path}")
        return module

    raise FileNotFoundError(
        "Could not find geometric_transformer/evaluate.py. Keep the release directory layout intact."
    )


def graph_config_from_cache_dir(cache_dir: Path) -> GraphBuildConfig:
    """Infer graph config from the standard cache tag when possible."""
    name = cache_dir.name

    # Standard tag: graph_v1_r6_n32 or v1_r6_n32.
    import re

    match = re.search(r"v(\d+)_r([0-9.]+)_n(\d+)", name)
    if match:
        return GraphBuildConfig(
            cache_schema_version=int(match.group(1)),
            radius=float(match.group(2)),
            max_neighbors=int(match.group(3)),
        )

    # Fall back to code defaults.
    return GraphBuildConfig()


def finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def finite_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.median()) if len(values) else np.nan


def common_atom_lddt(
    predicted: np.ndarray,
    sample: dict[str, Any],
    main_eval,
    radius: float,
) -> tuple[float, int]:
    """Diagnostic only: lDDT on atoms existing in WT and target.

    Mutant-only atoms (`is_new_atom`) are excluded because a literal WT copy has
    no prediction for them.
    """
    target = sample["pos_target"].cpu().numpy().astype(np.float64)

    observed = sample["target_mask"].cpu().numpy().astype(bool)
    new_atom = sample["is_new_atom"].cpu().numpy().astype(bool)
    residue = sample["node_residue"].cpu().numpy()

    common = observed & ~new_atom
    indices = np.where(common)[0]

    if len(indices) < 2:
        return np.nan, 0

    target_obs = target[indices]
    tree = cKDTree(target_obs)
    pairs = tree.query_pairs(r=float(radius), output_type="ndarray")

    if pairs.size == 0:
        return np.nan, 0

    global_i = indices[pairs[:, 0]]
    global_j = indices[pairs[:, 1]]

    # Match the main evaluator: exclude same-residue distances.
    keep = residue[global_i] != residue[global_j]
    global_i = global_i[keep]
    global_j = global_j[keep]

    if len(global_i) == 0:
        return np.nan, 0

    target_distance = np.linalg.norm(
        target[global_i] - target[global_j],
        axis=1,
    )
    predicted_distance = np.linalg.norm(
        predicted[global_i] - predicted[global_j],
        axis=1,
    )

    delta = np.abs(predicted_distance - target_distance)

    score = float(
        np.mean(
            [
                np.mean(delta < threshold)
                for threshold in (0.5, 1.0, 2.0, 4.0)
            ]
        )
    )
    return score, int(len(delta))


def common_sidechain_rmsd(
    predicted_aligned: np.ndarray,
    sample: dict[str, Any],
    main_eval,
) -> tuple[float, int]:
    """Diagnostic only: mutation-site side-chain RMSD for shared WT/MT atoms."""
    target = sample["pos_target"].cpu().numpy().astype(np.float64)
    observed = sample["target_mask"].cpu().numpy().astype(bool)
    new_atom = sample["is_new_atom"].cpu().numpy().astype(bool)
    mutation = sample["is_mutation_site"].cpu().numpy().astype(bool)

    names = np.asarray(
        main_eval.atom_names_for_sample(sample),
        dtype=object,
    )

    mask = (
        observed
        & ~new_atom
        & mutation
        & ~np.isin(names, list(main_eval.BACKBONE_NAMES))
    )

    n = int(mask.sum())
    if n == 0:
        return np.nan, 0

    diff = predicted_aligned[mask] - target[mask]
    rmsd = float(
        np.sqrt(
            np.mean(
                np.sum(diff * diff, axis=-1)
            )
        )
    )
    return rmsd, n


def copy_wt_metrics(
    sample: dict[str, Any],
    main_eval,
    local_radius: float,
    lddt_radius: float,
) -> dict[str, Any]:
    """Evaluate a literal unmodified WT/reference prediction.

    `pos_init` preserves WT coordinates for all atoms already present in the
    reference. Mutant-only atoms are explicitly excluded from Copy-WT scoring.
    Cα atoms are always shared, so the three backbone metrics are exact Copy-WT
    metrics.
    """
    pos_init = sample["pos_init"].cpu().numpy().astype(np.float64)

    # The three backbone metrics are unaffected by target-only side-chain atoms.
    full = main_eval.compute_metrics(
        predicted=pos_init,
        sample=sample,
        local_radius=local_radius,
        lddt_radius=lddt_radius,
    )

    # For literal Copy-WT, DO NOT report model-style full all-atom lDDT or
    # mutant-side-chain RMSD because pos_init contains constructed mutant-only
    # atoms. Those atoms are not part of the WT structure.
    pred_aligned, _ = main_eval.globally_align_prediction_to_target(
        pos_init,
        sample,
    )
    pred_aligned = main_eval.symmetry_correct_prediction(
        pred_aligned,
        sample,
    )

    diagnostic_lddt, diagnostic_lddt_pairs = common_atom_lddt(
        pred_aligned,
        sample,
        main_eval,
        radius=lddt_radius,
    )
    diagnostic_sc, diagnostic_sc_atoms = common_sidechain_rmsd(
        pred_aligned,
        sample,
        main_eval,
    )

    return {
        "ca_rmsd": full["ca_rmsd"],
        # Intentionally undefined in the main five-metric table.
        "lddt": np.nan,
        "local_ca_rmsd_8a": full["local_ca_rmsd_8a"],
        "mutation_sidechain_rmsd": np.nan,
        "mutation_response_rmsd_8a": full[
            "mutation_response_rmsd_8a"
        ],
        # Diagnostics saved separately:
        "copy_wt_common_atom_lddt": diagnostic_lddt,
        "copy_wt_common_atom_lddt_pairs": diagnostic_lddt_pairs,
        "copy_wt_common_sidechain_rmsd": diagnostic_sc,
        "copy_wt_common_sidechain_atom_count": diagnostic_sc_atoms,
        "common_ca_count": full["common_ca_count"],
        "local_ca_count_8a": full["local_ca_count_8a"],
    }


def mutant_init_metrics(
    sample: dict[str, Any],
    main_eval,
    local_radius: float,
    lddt_radius: float,
) -> dict[str, Any]:
    """No-learning baseline using the exact benchmark mutant initialization."""
    pos_init = sample["pos_init"].cpu().numpy().astype(np.float64)

    return main_eval.compute_metrics(
        predicted=pos_init,
        sample=sample,
        local_radius=local_radius,
        lddt_radius=lddt_radius,
    )


def event_macro(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    """Average repeated coordinate observations within each mutation event."""
    numeric_cols = [
        c
        for c in pair_metrics.columns
        if c not in {
            "baseline",
            "pair_id",
            "event_id",
            "pair_class",
            "mutation",
            "reference_pdb_id",
            "target_pdb_id",
        }
        and pd.api.types.is_numeric_dtype(pair_metrics[c])
    ]

    return (
        pair_metrics
        .groupby(["baseline", "event_id"], as_index=False, sort=False)
        [numeric_cols]
        .mean(numeric_only=True)
    )


def summarize(
    frame: pd.DataFrame,
    level_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for baseline, group in frame.groupby("baseline", sort=False):
        row: dict[str, Any] = {
            "baseline": baseline,
            f"n_{level_name}": int(len(group)),
        }

        for metric in ALL_FIVE:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]

            row[f"{metric}_n"] = int(len(finite))
            row[f"{metric}_mean"] = (
                float(finite.mean()) if len(finite) else np.nan
            )
            row[f"{metric}_median"] = (
                float(finite.median()) if len(finite) else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def manuscript_rows(event_summary: pd.DataFrame) -> pd.DataFrame:
    """Compact rows that can be concatenated with the model table."""
    wanted = {
        "baseline": "Model",
        "ca_rmsd_mean": "Cα RMSD ↓",
        "lddt_mean": "lDDT ↑",
        "local_ca_rmsd_8a_mean": "Local Cα RMSD 8Å ↓",
        "mutation_sidechain_rmsd_mean": "Mutation side-chain RMSD ↓",
        "mutation_response_rmsd_8a_mean": "MR-RMSD 8Å ↓",
    }

    return event_summary[list(wanted)].rename(columns=wanted)


def main() -> None:
    args = parse_args()
    started = time.time()

    main_eval = load_main_evaluator()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    graph_config = graph_config_from_cache_dir(args.cache_dir)
    pair_class = None if args.pair_class == "all" else args.pair_class

    dataset = PairGraphDataset(
        manifest_path=args.manifest,
        structures_dir=args.structures_dir,
        cache_dir=args.cache_dir,
        split="test",
        graph_config=graph_config,
        local_radius=None,
        pair_class=pair_class,
        max_samples=args.max_cases,
    )

    print("=" * 82)
    print("MutaStruct trivial baselines")
    print("=" * 82)
    print(f"Test coordinate pairs: {len(dataset):,}")
    print(f"Local radius:          {args.local_radius:.1f} Å")
    print(f"Output:                {args.output_dir}")
    print("=" * 82)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index in tqdm(range(len(dataset)), desc="test pairs"):
        try:
            sample = dataset[index]

            meta = {
                "pair_id": str(sample.get("pair_id", "")),
                "event_id": str(sample.get("event_id", "")),
                "pair_class": str(sample.get("pair_class", "")),
                "reference_pdb_id": str(
                    sample.get("reference_pdb_id", "")
                ),
                "target_pdb_id": str(
                    sample.get("target_pdb_id", "")
                ),
                "mutation": str(sample.get("mutation", "")),
                "mutation_index_0": int(
                    sample.get("mutation_index_0", -1)
                ),
                "native_wt_mt_ca_rmsd": float(
                    sample.get(
                        "target_alignment_ca_rmsd",
                        np.nan,
                    )
                ),
            }

            copy_metrics = copy_wt_metrics(
                sample,
                main_eval,
                local_radius=args.local_radius,
                lddt_radius=args.lddt_radius,
            )
            rows.append(
                {
                    "baseline": "Copy-WT",
                    **meta,
                    **copy_metrics,
                }
            )

            init_metrics = mutant_init_metrics(
                sample,
                main_eval,
                local_radius=args.local_radius,
                lddt_radius=args.lddt_radius,
            )
            rows.append(
                {
                    "baseline": "Mutant-Init",
                    **meta,
                    **init_metrics,
                }
            )

        except Exception as exc:
            errors.append(
                {
                    "index": index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    pair = pd.DataFrame(rows)

    if pair.empty:
        raise RuntimeError("No baseline metrics were computed")

    pair.to_csv(
        args.output_dir / "pair_metrics.csv",
        index=False,
    )

    events = event_macro(pair)
    events.to_csv(
        args.output_dir / "event_metrics.csv",
        index=False,
    )

    pair_summary = summarize(pair, "pairs")
    event_summary = summarize(events, "events")

    pair_summary.to_csv(
        args.output_dir / "summary_pair_macro.csv",
        index=False,
    )
    event_summary.to_csv(
        args.output_dir / "summary_event_macro.csv",
        index=False,
    )

    main_rows = manuscript_rows(event_summary)
    main_rows.to_csv(
        args.output_dir / "main_table_rows_event_macro.csv",
        index=False,
    )

    # Copy-WT diagnostics kept explicitly separate to avoid metric-definition
    # confusion in the main comparison table.
    diagnostics = pair[
        pair["baseline"].eq("Copy-WT")
    ][
        [
            "pair_id",
            "event_id",
            "pair_class",
            "copy_wt_common_atom_lddt",
            "copy_wt_common_atom_lddt_pairs",
            "copy_wt_common_sidechain_rmsd",
            "copy_wt_common_sidechain_atom_count",
        ]
    ].copy()

    diagnostics.to_csv(
        args.output_dir / "copy_wt_common_atom_diagnostics.csv",
        index=False,
    )

    if errors:
        pd.DataFrame(errors).to_csv(
            args.output_dir / "errors.csv",
            index=False,
        )

    metadata = {
        "test_pairs_requested": int(len(dataset)),
        "metric_protocol": {
            "Copy-WT": {
                "definition": (
                    "Unmodified experimental WT/reference coordinates."
                ),
                "main_comparable_metrics": [
                    "ca_rmsd",
                    "local_ca_rmsd_8a",
                    "mutation_response_rmsd_8a",
                ],
                "lddt_main_table": None,
                "mutation_sidechain_rmsd_main_table": None,
                "reason_for_na": (
                    "A literal WT structure has no coordinates for "
                    "mutant-only side-chain atoms."
                ),
            },
            "Mutant-Init": {
                "definition": (
                    "Exact chemically plausible mutant initialization "
                    "used as benchmark model input, with no learned refinement."
                ),
                "main_comparable_metrics": ALL_FIVE,
            },
        },
        "local_radius_angstrom": args.local_radius,
        "lddt_radius_angstrom": args.lddt_radius,
        "elapsed_seconds": time.time() - started,
    }

    (args.output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 82)
    print("EVENT-MACRO MEAN")
    print("=" * 82)

    display = main_rows.copy()
    for column in display.columns:
        if column != "Model":
            display[column] = pd.to_numeric(
                display[column],
                errors="coerce",
            ).map(
                lambda x: f"{x:.4f}" if np.isfinite(x) else "NA"
            )

    print(display.to_string(index=False))

    print()
    print("Interpretation:")
    print(
        "  Copy-WT: only Cα RMSD, Local Cα RMSD8Å, and MR-RMSD8Å "
        "are directly comparable in the main table."
    )
    print(
        "  Mutant-Init: all five metrics are directly comparable to benchmark."
    )

    print()
    print(f"Results: {args.output_dir}")
    if errors:
        print(f"Errors:  {args.output_dir / 'errors.csv'}")
    print("=" * 82)


if __name__ == "__main__":
    main()
