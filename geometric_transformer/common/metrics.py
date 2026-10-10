from __future__ import annotations

import numpy as np
import torch

from .constants import AA_ORDER, ATOM_NAMES, BACKBONE_ATOMS
from .geometry import kabsch_mobile_to_reference


_SIMPLE_SYMMETRY = {
    "D": [(("OD1", "OD2"),)],
    "E": [(("OE1", "OE2"),)],
    "R": [(("NH1", "NH2"),)],
    "L": [(("CD1", "CD2"),)],
    "V": [(("CG1", "CG2"),)],
}
_COUPLED_SYMMETRY = {
    "F": [(("CD1", "CD2"), ("CE1", "CE2"))],
    "Y": [(("CD1", "CD2"), ("CE1", "CE2"))],
}
_SIDECHAIN_BACKBONE = set(BACKBONE_ATOMS) | {"OXT"}


def _aligned_coordinates(
    mobile: np.ndarray,
    reference: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, float]:
    rotation, translation, rmsd = kabsch_mobile_to_reference(
        mobile,
        reference,
    )
    return points @ rotation + translation, rmsd


def _plain_rmsd(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    if len(predicted) == 0:
        return float("nan")

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    (predicted - target) ** 2,
                    axis=1,
                )
            )
        )
    )


def _atom_name(atom_type_value: int) -> str:
    if 0 <= int(atom_type_value) < len(ATOM_NAMES):
        return ATOM_NAMES[int(atom_type_value)]
    return "UNK"


def _mutation_sidechain_rmsd(
    predicted_aligned: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    is_mutation_site: np.ndarray,
    atom_type: np.ndarray,
    res_type_mt: np.ndarray,
) -> float:
    mutation_indices = np.where(
        target_mask
        & is_mutation_site
    )[0]

    if len(mutation_indices) == 0:
        return float("nan")

    names = {
        int(index): _atom_name(
            int(atom_type[index])
        )
        for index in mutation_indices
    }

    sidechain_indices = np.asarray(
        [
            index
            for index in mutation_indices
            if (
                names[int(index)]
                not in _SIDECHAIN_BACKBONE
                and names[int(index)] != "UNK"
            )
        ],
        dtype=int,
    )

    if len(sidechain_indices) == 0:
        return float("nan")

    identity = _plain_rmsd(
        predicted_aligned[sidechain_indices],
        target[sidechain_indices],
    )

    candidates = [identity]

    aa_id = int(
        res_type_mt[
            mutation_indices[0]
        ]
    )
    aa = (
        AA_ORDER[aa_id]
        if 0 <= aa_id < len(AA_ORDER)
        else None
    )

    if aa is None:
        return identity

    by_name = {
        names[int(index)]: int(index)
        for index in mutation_indices
        if names[int(index)] != "UNK"
    }

    permutations = []
    permutations.extend(
        _SIMPLE_SYMMETRY.get(aa, [])
    )
    permutations.extend(
        _COUPLED_SYMMETRY.get(aa, [])
    )

    local_lookup = {
        int(global_index): local_index
        for local_index, global_index
        in enumerate(sidechain_indices.tolist())
    }

    for swaps in permutations:
        if not all(
            a in by_name and b in by_name
            for a, b in swaps
        ):
            continue

        target_permuted = target[
            sidechain_indices
        ].copy()

        valid = True

        for name_a, name_b in swaps:
            ia = by_name[name_a]
            ib = by_name[name_b]

            if (
                ia not in local_lookup
                or ib not in local_lookup
            ):
                valid = False
                break

            la = local_lookup[ia]
            lb = local_lookup[ib]

            original_a = (
                target_permuted[la].copy()
            )
            original_b = (
                target_permuted[lb].copy()
            )

            target_permuted[la] = (
                original_b
            )
            target_permuted[lb] = (
                original_a
            )

        if valid:
            candidates.append(
                _plain_rmsd(
                    predicted_aligned[
                        sidechain_indices
                    ],
                    target_permuted,
                )
            )

    finite = [
        value
        for value in candidates
        if np.isfinite(value)
    ]

    return (
        float(min(finite))
        if finite
        else float("nan")
    )


def batch_structure_metrics(
    predicted: torch.Tensor,
    batch,
    local_radius: float = 8.0,
) -> dict[str, float]:
    """Per-graph validation metrics using a single global CA fit.

    Existing metrics are preserved and response-focused metrics are added:
      * local_ca_rmsd
      * mr_rmsd_8a (name remains fixed for logging compatibility)
      * copy_wt_mr_rmsd_8a
      * mr_excess_8a = model MR-RMSD - Copy-WT MR-RMSD
      * mutation_sidechain_rmsd

    Negative mr_excess_8a means the model beats Copy-WT on the response metric.
    """
    predicted_np = (
        predicted.detach()
        .float()
        .cpu()
        .numpy()
    )
    initial_np = (
        batch.pos_init.detach()
        .float()
        .cpu()
        .numpy()
    )
    target_np = (
        batch.pos_target.detach()
        .float()
        .cpu()
        .numpy()
    )
    target_mask = (
        batch.target_mask.detach()
        .cpu()
        .numpy()
        .astype(bool)
    )
    is_ca = (
        batch.is_ca.detach()
        .cpu()
        .numpy()
        .astype(bool)
    )
    distance = (
        batch.distance_to_mutation.detach()
        .float()
        .cpu()
        .numpy()
    )
    graph_ptr = (
        batch.graph_ptr.detach()
        .cpu()
        .numpy()
    )
    is_mutation_site = (
        batch.is_mutation_site.detach()
        .cpu()
        .numpy()
        .astype(bool)
    )
    atom_type = (
        batch.atom_type.detach()
        .cpu()
        .numpy()
    )
    res_type_mt = (
        batch.res_type_mt.detach()
        .cpu()
        .numpy()
    )

    ca_rmsd: list[float] = []
    init_ca_rmsd: list[float] = []
    local_rmsd: list[float] = []
    local_ca_rmsd: list[float] = []
    all_atom_rmsd: list[float] = []

    mr_rmsd: list[float] = []
    copy_mr_rmsd: list[float] = []
    mr_excess: list[float] = []
    sidechain_rmsd: list[float] = []

    for graph_index in range(
        len(graph_ptr) - 1
    ):
        start = int(
            graph_ptr[graph_index]
        )
        end = int(
            graph_ptr[graph_index + 1]
        )

        local_slice = slice(
            start,
            end,
        )

        mask = target_mask[
            local_slice
        ]

        ca = (
            mask
            & is_ca[
                local_slice
            ]
        )

        local = (
            mask
            & (
                distance[
                    local_slice
                ]
                <= float(local_radius)
            )
        )

        local_ca = (
            ca
            & (
                distance[
                    local_slice
                ]
                <= float(local_radius)
            )
        )

        pred_graph = predicted_np[
            local_slice
        ]
        init_graph = initial_np[
            local_slice
        ]
        target_graph = target_np[
            local_slice
        ]

        if int(ca.sum()) < 3:
            continue

        # Absolute target-structure metrics.
        pred_aligned, pred_ca_rmsd = (
            _aligned_coordinates(
                pred_graph[ca],
                target_graph[ca],
                pred_graph,
            )
        )

        init_aligned, init_rmsd = (
            _aligned_coordinates(
                init_graph[ca],
                target_graph[ca],
                init_graph,
            )
        )

        ca_rmsd.append(
            pred_ca_rmsd
        )
        init_ca_rmsd.append(
            init_rmsd
        )

        if int(local.sum()) > 0:
            local_rmsd.append(
                _plain_rmsd(
                    pred_aligned[
                        local
                    ],
                    target_graph[
                        local
                    ],
                )
            )

        if int(local_ca.sum()) > 0:
            local_ca_rmsd.append(
                _plain_rmsd(
                    pred_aligned[
                        local_ca
                    ],
                    target_graph[
                        local_ca
                    ],
                )
            )

        if int(mask.sum()) > 0:
            all_atom_rmsd.append(
                _plain_rmsd(
                    pred_aligned[
                        mask
                    ],
                    target_graph[
                        mask
                    ],
                )
            )

        sidechain_rmsd.append(
            _mutation_sidechain_rmsd(
                predicted_aligned=pred_aligned,
                target=target_graph,
                target_mask=mask,
                is_mutation_site=is_mutation_site[
                    local_slice
                ],
                atom_type=atom_type[
                    local_slice
                ],
                res_type_mt=res_type_mt[
                    local_slice
                ],
            )
        )

        # Response metric:
        # independently align predicted and native target CA coordinates back
        # into the WT/init frame, then compare local CA displacement vectors.
        pred_in_wt_frame, _ = (
            _aligned_coordinates(
                pred_graph[ca],
                init_graph[ca],
                pred_graph,
            )
        )

        target_in_wt_frame, _ = (
            _aligned_coordinates(
                target_graph[ca],
                init_graph[ca],
                target_graph,
            )
        )

        if int(local_ca.sum()) > 0:
            delta_pred = (
                pred_in_wt_frame[
                    local_ca
                ]
                - init_graph[
                    local_ca
                ]
            )

            delta_native = (
                target_in_wt_frame[
                    local_ca
                ]
                - init_graph[
                    local_ca
                ]
            )

            model_mr = _plain_rmsd(
                delta_pred,
                delta_native,
            )

            copy_mr = _plain_rmsd(
                np.zeros_like(
                    delta_native
                ),
                delta_native,
            )

            mr_rmsd.append(
                model_mr
            )
            copy_mr_rmsd.append(
                copy_mr
            )
            mr_excess.append(
                model_mr
                - copy_mr
            )

    def mean_or_nan(
        values: list[float],
    ) -> float:
        finite = [
            value
            for value in values
            if np.isfinite(value)
        ]

        return (
            float(
                np.mean(finite)
            )
            if finite
            else float("nan")
        )

    return {
        # Existing names.
        "ca_rmsd": mean_or_nan(
            ca_rmsd
        ),
        "init_ca_rmsd": mean_or_nan(
            init_ca_rmsd
        ),
        "local_rmsd": mean_or_nan(
            local_rmsd
        ),
        "all_atom_rmsd": mean_or_nan(
            all_atom_rmsd
        ),

        # Response-focused metrics.
        "local_ca_rmsd": mean_or_nan(
            local_ca_rmsd
        ),
        "mr_rmsd_8a": mean_or_nan(
            mr_rmsd
        ),
        "copy_wt_mr_rmsd_8a": (
            mean_or_nan(
                copy_mr_rmsd
            )
        ),
        "mr_excess_8a": mean_or_nan(
            mr_excess
        ),
        "mutation_sidechain_rmsd": (
            mean_or_nan(
                sidechain_rmsd
            )
        ),
    }


def batch_response_metrics_per_graph(
    predicted: torch.Tensor,
    batch,
    local_radius: float = 8.0,
) -> list[dict[str, float]]:
    """Return response-sensitive validation metrics separately for each graph.

    This is used by the trainer to compute event-macro validation monitors,
    matching the benchmark's primary aggregation more closely than pair-macro
    checkpoint selection.
    """
    predicted_np = (
        predicted.detach().float().cpu().numpy()
    )
    initial_np = (
        batch.pos_init.detach().float().cpu().numpy()
    )
    target_np = (
        batch.pos_target.detach().float().cpu().numpy()
    )
    target_mask = (
        batch.target_mask.detach().cpu().numpy().astype(bool)
    )
    is_ca = (
        batch.is_ca.detach().cpu().numpy().astype(bool)
    )
    distance = (
        batch.distance_to_mutation.detach().float().cpu().numpy()
    )
    graph_ptr = (
        batch.graph_ptr.detach().cpu().numpy()
    )
    is_mutation_site = (
        batch.is_mutation_site.detach().cpu().numpy().astype(bool)
    )
    atom_type = (
        batch.atom_type.detach().cpu().numpy()
    )
    res_type_mt = (
        batch.res_type_mt.detach().cpu().numpy()
    )

    results: list[dict[str, float]] = []

    for graph_index in range(len(graph_ptr) - 1):
        start = int(graph_ptr[graph_index])
        end = int(graph_ptr[graph_index + 1])
        sl = slice(start, end)

        mask = target_mask[sl]
        ca = mask & is_ca[sl]
        local_ca = ca & (
            distance[sl] <= float(local_radius)
        )

        pred_graph = predicted_np[sl]
        init_graph = initial_np[sl]
        target_graph = target_np[sl]

        graph_result = {
            "local_ca_rmsd": float("nan"),
            "mr_rmsd_8a": float("nan"),
            "copy_wt_mr_rmsd_8a": float("nan"),
            "mr_excess_8a": float("nan"),
            "mutation_sidechain_rmsd": float("nan"),
        }

        if int(ca.sum()) < 3:
            results.append(graph_result)
            continue

        pred_target_aligned, _ = _aligned_coordinates(
            pred_graph[ca],
            target_graph[ca],
            pred_graph,
        )

        if int(local_ca.sum()) > 0:
            graph_result["local_ca_rmsd"] = _plain_rmsd(
                pred_target_aligned[local_ca],
                target_graph[local_ca],
            )

        graph_result["mutation_sidechain_rmsd"] = (
            _mutation_sidechain_rmsd(
                predicted_aligned=pred_target_aligned,
                target=target_graph,
                target_mask=mask,
                is_mutation_site=is_mutation_site[sl],
                atom_type=atom_type[sl],
                res_type_mt=res_type_mt[sl],
            )
        )

        pred_wt_aligned, _ = _aligned_coordinates(
            pred_graph[ca],
            init_graph[ca],
            pred_graph,
        )
        target_wt_aligned, _ = _aligned_coordinates(
            target_graph[ca],
            init_graph[ca],
            target_graph,
        )

        if int(local_ca.sum()) > 0:
            delta_pred = (
                pred_wt_aligned[local_ca]
                - init_graph[local_ca]
            )
            delta_native = (
                target_wt_aligned[local_ca]
                - init_graph[local_ca]
            )

            model_mr = _plain_rmsd(
                delta_pred,
                delta_native,
            )
            copy_mr = _plain_rmsd(
                np.zeros_like(delta_native),
                delta_native,
            )

            graph_result["mr_rmsd_8a"] = model_mr
            graph_result["copy_wt_mr_rmsd_8a"] = copy_mr
            graph_result["mr_excess_8a"] = model_mr - copy_mr

        results.append(graph_result)

    return results
