#!/usr/bin/env python3
"""Evaluate the sparse Qin-inspired GeoTransformer on the locked Tier-A test set.

Primary metrics
---------------
1. Global Cα RMSD ↓
   Predicted mutant is globally superposed onto the experimental mutant using all
   mutually represented/observed Cα atoms.

2. All-atom lDDT ↑
   Superposition-free all-atom lDDT with the standard 15 Å inclusion radius and
   distance tolerances {0.5, 1, 2, 4} Å. Only heavy atoms represented by the model
   graph and observed in the experimental mutant are scored; same-residue atom
   pairs are excluded. Common symmetric side-chain naming ambiguities are corrected
   before scoring.

3. Local Cα RMSD (8 Å) ↓
   Uses the SAME global Cα superposition as metric 1, then scores Cα atoms whose WT
   Cα positions lie within 8 Å of the mutation-site Cα. The local region is therefore
   fixed from the WT/reference structure and is identical for every model.

4. Mutation-site side-chain RMSD ↓
   Heavy side-chain atoms at the substituted residue after the SAME global Cα
   superposition. Backbone N/CA/C/O/OXT atoms are excluded. Symmetry-equivalent atom
   naming is corrected (e.g. ASP OD1/OD2, LEU CD1/CD2).

5. Mutation-response RMSD (8 Å) ↓
   Measures whether the model reproduces the experimentally observed WT→mutant
   backbone displacement field. Prediction and experimental mutant are each placed
   in the WT frame using the global common-Cα fit, and the RMSD between their Cα
   displacement vectors is computed within the fixed 8 Å WT neighborhood.

Why event-macro?
----------------
Multiple coordinate pairs may correspond to one biological mutation event. The
script therefore saves:
  * pair-level metrics,
  * pair-macro summaries, and
  * event-macro summaries (recommended for the manuscript main table).

The model is evaluated fairly on the full protein: it is run only on its
mutation-centered crop, and atoms outside the crop are copied unchanged from the
mutant initialization before global metrics are computed.

Example
-------
Smoke test:
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --max-cases 32 --amp

Full test evaluation:
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --amp
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.constants import (  # noqa: E402
    AA_ORDER,
    ATOM_NAMES,
    DEFAULT_CACHE_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_STRUCTURES_DIR,
)
from common.data import (  # noqa: E402
    PairGraphDataset,
    collate_graph_samples,
    crop_graph_sample,
)
from common.geometry import kabsch_mobile_to_reference  # noqa: E402
from common.graph import GraphBuildConfig  # noqa: E402
from network.config import ModelConfig  # noqa: E402
from network.model import GeometricTransformer  # noqa: E402


DEFAULT_OUTPUT = ROOT.parent / "results" / "geometric_transformer"

# Backbone atoms are excluded from mutation-site side-chain RMSD.
BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT"}

# Chemically equivalent atom-name permutations that should not be penalized.
# PHE/TYR ring swaps are coupled.
SYMMETRY_SWAPS: dict[str, dict[str, str]] = {
    "D": {"OD1": "OD2", "OD2": "OD1"},
    "E": {"OE1": "OE2", "OE2": "OE1"},
    "R": {"NH1": "NH2", "NH2": "NH1"},
    "L": {"CD1": "CD2", "CD2": "CD1"},
    "V": {"CG1": "CG2", "CG2": "CG1"},
    "F": {
        "CD1": "CD2",
        "CD2": "CD1",
        "CE1": "CE2",
        "CE2": "CE1",
    },
    "Y": {
        "CD1": "CD2",
        "CD2": "CD1",
        "CE1": "CE2",
        "CE2": "CE1",
    },
}

PRIMARY_METRICS = [
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
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT.parent
            / "checkpoints"
            / "geometric_transformer_best.pt"
        ),
    )

    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--pair-class", default="all")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--local-radius",
        type=float,
        default=8.0,
        help="WT C-alpha neighborhood radius for local and response RMSD.",
    )
    p.add_argument(
        "--lddt-radius",
        type=float,
        default=15.0,
        help="Reference distance inclusion radius for all-atom lDDT.",
    )
    return p.parse_args()

def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=bool(enabled and device.type == "cuda"),
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def config_from_checkpoint(config_class, checkpoint: dict[str, Any]):
    cfg = config_class()
    saved = checkpoint.get("configuration", {}).get("model", {})
    valid = {field.name for field in fields(cfg)}
    for key, value in saved.items():
        if key in valid:
            setattr(cfg, key, value)
    return cfg


def build_models(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], float, GraphBuildConfig, dict[str, Any]]:
    checkpoint = load_checkpoint(args.checkpoint)
    config = config_from_checkpoint(ModelConfig, checkpoint)
    model = GeometricTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    graph_saved = dict(
        checkpoint.get("configuration", {}).get("graph", {})
    )
    graph_config = GraphBuildConfig(
        radius=float(graph_saved.get("radius", 6.0)),
        max_neighbors=int(graph_saved.get("max_neighbors", 32)),
        cache_schema_version=int(
            graph_saved.get("cache_schema_version", 1)
        ),
    )
    crop_radius = float(
        checkpoint.get("configuration", {})
        .get("model", {})
        .get("crop_radius", getattr(config, "crop_radius", 12.0))
    )
    metadata = {
        "Geometric Transformer": {
            "checkpoint": str(args.checkpoint),
            "epoch": int(checkpoint.get("epoch", -1)),
            "saved_validation_metrics": checkpoint.get("metrics", {}),
            "model_config": checkpoint.get("configuration", {}).get(
                "model", {}
            ),
            "graph_config": graph_saved,
        }
    }
    return (
        {"Geometric Transformer": model},
        crop_radius,
        graph_config,
        metadata,
    )

def _fit_transform(
    mobile: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(mobile) < 3:
        raise ValueError("At least three Cα atoms are required for global superposition")
    return kabsch_mobile_to_reference(
        np.asarray(mobile, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    )


def _apply_transform(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ rotation + translation


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


def common_ca_mask(sample: dict[str, Any]) -> np.ndarray:
    return (
        sample["target_mask"].cpu().numpy().astype(bool)
        & sample["is_ca"].cpu().numpy().astype(bool)
    )


def globally_align_prediction_to_target(
    predicted: np.ndarray,
    sample: dict[str, Any],
) -> tuple[np.ndarray, float]:
    target = sample["pos_target"].cpu().numpy().astype(np.float64)
    ca = common_ca_mask(sample)
    if int(ca.sum()) < 3:
        raise ValueError(f"{sample['pair_id']}: fewer than three observed common Cα atoms")
    rotation, translation, fit_rmsd = _fit_transform(
        predicted[ca],
        target[ca],
    )
    return _apply_transform(predicted, rotation, translation), float(fit_rmsd)


def target_aa_at_mutation(sample: dict[str, Any]) -> str:
    mutation = sample["is_mutation_site"].cpu().numpy().astype(bool)
    ids = sample["res_type_mt"].cpu().numpy()
    if not mutation.any():
        return "X"
    aa_id = int(ids[np.where(mutation)[0][0]])
    if 0 <= aa_id < len(AA_ORDER):
        return AA_ORDER[aa_id]
    return "X"


def atom_names_for_sample(sample: dict[str, Any]) -> list[str]:
    ids = sample["atom_type"].cpu().numpy()
    output: list[str] = []
    for atom_id in ids:
        atom_id = int(atom_id)
        if 0 <= atom_id < len(ATOM_NAMES):
            output.append(ATOM_NAMES[atom_id])
        else:
            output.append("UNK")
    return output


def symmetry_correct_prediction(
    predicted_aligned: np.ndarray,
    sample: dict[str, Any],
) -> np.ndarray:
    """Swap symmetry-equivalent predicted atom labels where this improves fit.

    This is a scoring-only correction for conventional PDB atom-name ambiguity.
    It is never fed back into the model.
    """
    result = np.asarray(predicted_aligned, dtype=np.float64).copy()
    target = sample["pos_target"].cpu().numpy().astype(np.float64)
    observed = sample["target_mask"].cpu().numpy().astype(bool)
    residue_ids = sample["node_residue"].cpu().numpy()
    mt_ids = sample["res_type_mt"].cpu().numpy()
    names = atom_names_for_sample(sample)

    for residue_id in np.unique(residue_ids):
        idx = np.where(residue_ids == residue_id)[0]
        if len(idx) == 0:
            continue

        aa_id = int(mt_ids[idx[0]])
        if not (0 <= aa_id < len(AA_ORDER)):
            continue
        aa = AA_ORDER[aa_id]
        swap = SYMMETRY_SWAPS.get(aa)
        if not swap:
            continue

        name_to_index = {
            names[i]: int(i)
            for i in idx
            if observed[i]
        }
        swap_names = sorted(swap)
        if not all(name in name_to_index for name in swap_names):
            continue

        current_indices = [name_to_index[name] for name in swap_names]
        source_indices = [name_to_index[swap[name]] for name in swap_names]

        identity_error = float(
            np.sum(
                (
                    result[current_indices]
                    - target[current_indices]
                ) ** 2
            )
        )
        swapped_error = float(
            np.sum(
                (
                    result[source_indices]
                    - target[current_indices]
                ) ** 2
            )
        )

        if swapped_error < identity_error:
            old = result.copy()
            for target_name in swap_names:
                dst = name_to_index[target_name]
                src = name_to_index[swap[target_name]]
                result[dst] = old[src]

    return result


def all_atom_lddt(
    predicted: np.ndarray,
    sample: dict[str, Any],
    radius: float = 15.0,
) -> tuple[float, int]:
    """Standard all-atom lDDT on represented, target-observed heavy atoms.

    Reference pairs are target atom pairs closer than `radius`, excluding atoms
    belonging to the same residue. Score is the mean preserved-distance fraction
    at 0.5, 1, 2, and 4 Å.
    """
    target = sample["pos_target"].cpu().numpy().astype(np.float64)
    observed = sample["target_mask"].cpu().numpy().astype(bool)
    residue = sample["node_residue"].cpu().numpy()

    indices = np.where(observed)[0]
    if len(indices) < 2:
        return float("nan"), 0

    target_obs = target[indices]
    tree = cKDTree(target_obs)
    pairs = tree.query_pairs(r=float(radius), output_type="ndarray")
    if pairs.size == 0:
        return float("nan"), 0

    global_i = indices[pairs[:, 0]]
    global_j = indices[pairs[:, 1]]

    different_residue = residue[global_i] != residue[global_j]
    global_i = global_i[different_residue]
    global_j = global_j[different_residue]

    if len(global_i) == 0:
        return float("nan"), 0

    target_distance = np.linalg.norm(
        target[global_i] - target[global_j],
        axis=1,
    )
    predicted_distance = np.linalg.norm(
        predicted[global_i] - predicted[global_j],
        axis=1,
    )
    delta = np.abs(predicted_distance - target_distance)

    thresholds = (0.5, 1.0, 2.0, 4.0)
    score = float(
        np.mean(
            [
                np.mean(delta < threshold)
                for threshold in thresholds
            ]
        )
    )
    return score, int(len(delta))


def mutation_sidechain_rmsd(
    predicted_aligned: np.ndarray,
    sample: dict[str, Any],
) -> tuple[float, int]:
    target = sample["pos_target"].cpu().numpy().astype(np.float64)
    observed = sample["target_mask"].cpu().numpy().astype(bool)
    mutation = sample["is_mutation_site"].cpu().numpy().astype(bool)
    names = np.asarray(atom_names_for_sample(sample), dtype=object)

    sidechain = (
        observed
        & mutation
        & ~np.isin(names, list(BACKBONE_NAMES))
    )
    count = int(sidechain.sum())
    if count == 0:
        return float("nan"), 0

    return _rmsd(
        predicted_aligned[sidechain],
        target[sidechain],
    ), count


def mutation_response_rmsd(
    predicted: np.ndarray,
    sample: dict[str, Any],
    local_radius: float,
) -> tuple[float, int]:
    """Compare predicted vs native WT→MT Cα displacement vectors in the WT frame."""
    initial = sample["pos_init"].cpu().numpy().astype(np.float64)
    target = sample["pos_target"].cpu().numpy().astype(np.float64)
    ca = common_ca_mask(sample)
    distance = sample["distance_to_mutation"].cpu().numpy().astype(np.float64)
    local_ca = ca & (distance <= float(local_radius))

    if int(ca.sum()) < 3 or int(local_ca.sum()) == 0:
        return float("nan"), int(local_ca.sum())

    # Remove any global rigid-body drift from the prediction.
    pred_rotation, pred_translation, _ = _fit_transform(
        predicted[ca],
        initial[ca],
    )
    pred_wt_frame = _apply_transform(
        predicted,
        pred_rotation,
        pred_translation,
    )

    # The graph builder already aligns the experimental mutant to WT, but fit it
    # again here explicitly so the metric definition does not depend on that detail.
    target_rotation, target_translation, _ = _fit_transform(
        target[ca],
        initial[ca],
    )
    target_wt_frame = _apply_transform(
        target,
        target_rotation,
        target_translation,
    )

    delta_pred = pred_wt_frame[local_ca] - initial[local_ca]
    delta_native = target_wt_frame[local_ca] - initial[local_ca]

    return _rmsd(delta_pred, delta_native), int(local_ca.sum())


def compute_metrics(
    predicted: np.ndarray,
    sample: dict[str, Any],
    local_radius: float,
    lddt_radius: float,
) -> dict[str, Any]:
    target = sample["pos_target"].cpu().numpy().astype(np.float64)
    distance = sample["distance_to_mutation"].cpu().numpy().astype(np.float64)
    ca = common_ca_mask(sample)

    if int(ca.sum()) < 3:
        raise ValueError("Fewer than three common target-observed Cα atoms")

    pred_aligned, ca_rmsd = globally_align_prediction_to_target(
        predicted,
        sample,
    )

    # Resolve common chemically symmetric atom-name ambiguities for atom-level
    # scoring only. Cα metrics are unaffected.
    pred_sym = symmetry_correct_prediction(
        pred_aligned,
        sample,
    )

    local_ca = ca & (distance <= float(local_radius))
    local_count = int(local_ca.sum())
    local_ca_rmsd = (
        _rmsd(pred_aligned[local_ca], target[local_ca])
        if local_count > 0
        else float("nan")
    )

    # lDDT is rigid-body invariant, so using the globally aligned/symmetry-corrected
    # coordinates is equivalent with respect to rigid-body placement.
    lddt, lddt_pairs = all_atom_lddt(
        pred_sym,
        sample,
        radius=lddt_radius,
    )

    sidechain_rmsd, sidechain_count = mutation_sidechain_rmsd(
        pred_sym,
        sample,
    )

    response_rmsd, response_count = mutation_response_rmsd(
        predicted,
        sample,
        local_radius=local_radius,
    )

    return {
        "ca_rmsd": ca_rmsd,
        "lddt": lddt,
        "local_ca_rmsd_8a": local_ca_rmsd,
        "mutation_sidechain_rmsd": sidechain_rmsd,
        "mutation_response_rmsd_8a": response_rmsd,
        "common_ca_count": int(ca.sum()),
        "local_ca_count_8a": local_count,
        "sidechain_atom_count": sidechain_count,
        "lddt_distance_count": lddt_pairs,
        "response_ca_count_8a": response_count,
    }


@torch.inference_mode()
def predict_full_structure(
    model_name: str,
    model: torch.nn.Module,
    full_sample: dict[str, Any],
    device: torch.device,
    amp: bool,
    local_crop_radius: float,
) -> np.ndarray:
    """Run the local model and restore unchanged atoms outside the crop."""
    del model_name
    selected = (
        (full_sample["distance_to_mutation"] <= float(local_crop_radius))
        | full_sample["is_mutation_site"]
    )
    selected_indices = torch.where(selected)[0]

    cropped = crop_graph_sample(
        full_sample,
        radius=local_crop_radius,
    )
    batch = collate_graph_samples([cropped]).to(
        device,
        non_blocking=True,
    )
    with autocast_context(device, amp):
        output = model(batch)["pos"]

    full_prediction = full_sample["pos_init"].detach().clone()
    full_prediction[selected_indices] = output.detach().float().cpu()
    return full_prediction.numpy().astype(np.float64)

def summarize_frame(
    frame: pd.DataFrame,
    group_level: str,
) -> pd.DataFrame:
    """One row/model summary with mean, median, SD, and evaluable N."""
    rows: list[dict[str, Any]] = []

    for model, group in frame.groupby("model", sort=False):
        row: dict[str, Any] = {
            "model": model,
            f"n_{group_level}": int(len(group)),
        }
        for metric in PRIMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            row[f"{metric}_n"] = int(len(finite))
            row[f"{metric}_mean"] = (
                float(finite.mean()) if len(finite) else np.nan
            )
            row[f"{metric}_median"] = (
                float(finite.median()) if len(finite) else np.nan
            )
            row[f"{metric}_sd"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def manuscript_table(event_summary: pd.DataFrame) -> pd.DataFrame:
    """Compact event-macro table containing only the five primary mean metrics."""
    columns = {
        "model": "Model",
        "ca_rmsd_mean": "Cα RMSD ↓",
        "lddt_mean": "lDDT ↑",
        "local_ca_rmsd_8a_mean": "Local Cα RMSD 8Å ↓",
        "mutation_sidechain_rmsd_mean": "Mutation side-chain RMSD ↓",
        "mutation_response_rmsd_8a_mean": "MR-RMSD 8Å ↓",
    }
    table = event_summary[list(columns)].rename(columns=columns).copy()
    return table


def main() -> None:
    args = parse_args()
    started = time.time()

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models, local_crop_radius, graph_config, checkpoint_metadata = build_models(
        args,
        device,
    )

    pair_class = None if args.pair_class == "all" else args.pair_class

    # Always load the UNCROPPED test graph. The local model is cropped only for its forward
    # pass, then reconstructed to the full protein for fair global evaluation.
    dataset = PairGraphDataset(
        manifest_path=args.manifest,
        structures_dir=args.structures_dir,
        cache_dir=args.cache_dir,
        split=args.split,
        graph_config=graph_config,
        local_radius=None,
        pair_class=pair_class,
        max_samples=args.max_cases,
    )

    print("=" * 88)
    print("MutaStruct Geometric Transformer — Main-5 Evaluation")
    print("=" * 88)
    print(f"Device:               {device}")
    print(f"AMP:                  {args.amp and device.type == 'cuda'}")
    print(f"Split:                {args.split}")
    print(f"Coordinate pairs:     {len(dataset):,}")
    print(f"Local metric radius:  {args.local_radius:.1f} Å")
    print(f"lDDT radius:          {args.lddt_radius:.1f} Å")
    print(f"Model crop radius: {local_crop_radius:.1f} Å")
    print(f"Output:               {args.output_dir}")
    print("=" * 88)

    result_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    progress = tqdm(range(len(dataset)), desc=f"{args.split} pairs")

    for index in progress:
        try:
            sample = dataset[index]
        except Exception as exc:
            error_rows.append(
                {
                    "index": index,
                    "model": "ALL",
                    "pair_id": "",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        metadata = {
            "pair_id": str(sample.get("pair_id", "")),
            "event_id": str(sample.get("event_id", "")),
            "pair_class": str(sample.get("pair_class", "")),
            "reference_pdb_id": str(sample.get("reference_pdb_id", "")),
            "target_pdb_id": str(sample.get("target_pdb_id", "")),
            "mutation": str(sample.get("mutation", "")),
            "mutation_index_0": int(sample.get("mutation_index_0", -1)),
            "native_wt_mt_ca_rmsd": float(
                sample.get("target_alignment_ca_rmsd", np.nan)
            ),
            "target_aa": target_aa_at_mutation(sample),
        }

        for model_name, model in models.items():
            try:
                predicted = predict_full_structure(
                    model_name=model_name,
                    model=model,
                    full_sample=sample,
                    device=device,
                    amp=args.amp,
                    local_crop_radius=local_crop_radius,
                )

                metrics = compute_metrics(
                    predicted=predicted,
                    sample=sample,
                    local_radius=args.local_radius,
                    lddt_radius=args.lddt_radius,
                )

                result_rows.append(
                    {
                        "model": model_name,
                        **metadata,
                        **metrics,
                    }
                )

            except Exception as exc:
                error_rows.append(
                    {
                        "index": index,
                        "model": model_name,
                        "pair_id": metadata["pair_id"],
                        "event_id": metadata["event_id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    results = pd.DataFrame(result_rows)
    errors = pd.DataFrame(error_rows)

    results_path = args.output_dir / "pair_metrics.csv"
    results.to_csv(results_path, index=False)

    if len(errors):
        errors.to_csv(args.output_dir / "errors.csv", index=False)

    if results.empty:
        raise RuntimeError(
            "No successful predictions were evaluated. "
            f"See {args.output_dir / 'errors.csv'}"
        )

    # Pair-macro summary: each coordinate pair receives equal weight.
    pair_summary = summarize_frame(
        results,
        group_level="pairs",
    )
    pair_summary.to_csv(
        args.output_dir / "summary_pair_macro.csv",
        index=False,
    )

    # Event-macro summary: first average repeated coordinate observations within
    # each biological mutation event, then average events equally.
    event_metrics = (
        results.groupby(
            ["model", "event_id"],
            as_index=False,
            sort=False,
        )[PRIMARY_METRICS]
        .mean(numeric_only=True)
    )
    event_metrics.to_csv(
        args.output_dir / "event_metrics.csv",
        index=False,
    )

    event_summary = summarize_frame(
        event_metrics,
        group_level="events",
    )
    event_summary.to_csv(
        args.output_dir / "summary_event_macro.csv",
        index=False,
    )

    main_table = manuscript_table(event_summary)
    main_table.to_csv(
        args.output_dir / "main_table_event_macro.csv",
        index=False,
    )

    # Pair-class sensitivity analysis using event-macro values within each class.
    class_event = (
        results.groupby(
            ["model", "pair_class", "event_id"],
            as_index=False,
            sort=False,
        )[PRIMARY_METRICS]
        .mean(numeric_only=True)
    )
    class_rows: list[dict[str, Any]] = []
    for (model, pair_class_value), group in class_event.groupby(
        ["model", "pair_class"],
        sort=False,
    ):
        row: dict[str, Any] = {
            "model": model,
            "pair_class": pair_class_value,
            "n_events": int(group["event_id"].nunique()),
        }
        for metric in PRIMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = (
                float(finite.mean()) if len(finite) else np.nan
            )
            row[f"{metric}_median"] = (
                float(finite.median()) if len(finite) else np.nan
            )
            row[f"{metric}_n"] = int(len(finite))
        class_rows.append(row)

    pd.DataFrame(class_rows).to_csv(
        args.output_dir / "summary_by_pair_class_event_macro.csv",
        index=False,
    )

    metadata_out = {
        "split": args.split,
        "pairs_requested": len(dataset),
        "successful_metric_rows": int(len(results)),
        "errors": int(len(errors)),
        "primary_metrics": PRIMARY_METRICS,
        "recommended_primary_aggregation": "event-macro mean",
        "local_radius_angstrom": float(args.local_radius),
        "lddt_radius_angstrom": float(args.lddt_radius),
        "lddt_thresholds_angstrom": [0.5, 1.0, 2.0, 4.0],
        "model_crop_radius_angstrom": float(local_crop_radius),
        "manifest": str(args.manifest),
        "structures_dir": str(args.structures_dir),
        "cache_dir": str(args.cache_dir),
        "graph_config": {
            "radius": graph_config.radius,
            "max_neighbors": graph_config.max_neighbors,
            "cache_schema_version": graph_config.cache_schema_version,
        },
        "checkpoints": checkpoint_metadata,
        "metric_notes": {
            "ca_rmsd": (
                "Global C-alpha RMSD after optimal fit over all represented, "
                "target-observed common C-alpha atoms."
            ),
            "lddt": (
                "All-atom lDDT over represented, target-observed heavy atoms; "
                "15 A target inclusion radius; same-residue pairs excluded; "
                "0.5/1/2/4 A thresholds; common symmetric atom naming corrected."
            ),
            "local_ca_rmsd_8a": (
                "C-alpha RMSD inside the WT-defined 8 A mutation neighborhood "
                "after the same global C-alpha fit used for ca_rmsd."
            ),
            "mutation_sidechain_rmsd": (
                "Heavy-atom side-chain RMSD of the substituted residue after "
                "global C-alpha fit; backbone excluded; symmetry corrected."
            ),
            "mutation_response_rmsd_8a": (
                "RMSD between predicted and experimental WT-to-mutant C-alpha "
                "displacement vectors within the WT-defined 8 A neighborhood, "
                "after independently removing global rigid-body motion relative "
                "to WT."
            ),
        },
        "elapsed_seconds": float(time.time() - started),
    }

    (args.output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata_out, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("Recommended manuscript table: EVENT-MACRO MEAN")
    print("=" * 88)
    display = main_table.copy()
    for column in display.columns:
        if column != "Model":
            display[column] = pd.to_numeric(
                display[column], errors="coerce"
            ).map(lambda x: f"{x:.4f}" if np.isfinite(x) else "NA")
    print(display.to_string(index=False))

    print()
    print(f"Pair-level metrics:     {results_path}")
    print(f"Event-macro summary:    {args.output_dir / 'summary_event_macro.csv'}")
    print(f"Manuscript table:       {args.output_dir / 'main_table_event_macro.csv'}")
    if len(errors):
        print(f"Errors:                 {args.output_dir / 'errors.csv'}")
    print("=" * 88)


if __name__ == "__main__":
    main()
