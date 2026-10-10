from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from .constants import (
    AA_TO_ID,
    ATOM_TO_ID,
    ELEMENT_TO_ID,
    UNK_AA_ID,
    UNK_ATOM_ID,
)
from .geometry import dihedral_np, residue_frame_np, wrap_angle
from .structure import PairStructure, build_pair_structure
from .templates import residue_bonds


@dataclass(frozen=True)
class GraphBuildConfig:
    radius: float = 6.0
    max_neighbors: int = 32
    cache_schema_version: int = 1

    def tag(self) -> str:
        return (
            f"v{self.cache_schema_version}_r{self.radius:g}_n{self.max_neighbors}"
        )


def _element_id(element: str) -> int:
    normalized = str(element).strip().upper()
    if normalized not in ELEMENT_TO_ID:
        normalized = "OTHER"
    return ELEMENT_TO_ID[normalized]


def _target_coordinate_arrays(pair: PairStructure) -> tuple[np.ndarray, np.ndarray]:
    positions: list[np.ndarray] = []
    mask: list[bool] = []
    for node in pair.nodes:
        target = pair.target_positions.get((node.seq_id, node.atom_name))
        if target is None:
            positions.append(node.pos_init)
            mask.append(False)
        else:
            positions.append(target)
            mask.append(True)
    return np.asarray(positions, dtype=np.float32), np.asarray(mask, dtype=bool)


def _build_edges(
    positions: np.ndarray,
    seq_ids: np.ndarray,
    atom_names: list[str],
    target_sequence: str,
    mutation_position: int,
    config: GraphBuildConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build directed spatial edges plus covalent edges."""
    n_nodes = len(positions)
    lookup = {(int(seq_ids[i]), atom_names[i]): i for i in range(n_nodes)}
    edges: dict[tuple[int, int], bool] = {}

    def add_edge(i: int, j: int, covalent: bool) -> None:
        if i == j:
            return
        key = (int(i), int(j))
        edges[key] = bool(edges.get(key, False) or covalent)

    # Standard intra-residue covalent graph derived from ideal amino-acid templates.
    for seq_id in sorted(set(int(x) for x in seq_ids)):
        if not (1 <= seq_id <= len(target_sequence)):
            continue
        aa = target_sequence[seq_id - 1]
        include_oxt = seq_id == len(target_sequence)
        for atom_a, atom_b in residue_bonds(aa, include_oxt=include_oxt):
            key_a = (seq_id, atom_a)
            key_b = (seq_id, atom_b)
            if key_a in lookup and key_b in lookup:
                i, j = lookup[key_a], lookup[key_b]
                add_edge(i, j, True)
                add_edge(j, i, True)

    # Peptide C--N bonds between adjacent represented residues.
    represented = sorted(set(int(x) for x in seq_ids))
    for seq_id in represented:
        if (seq_id, "C") in lookup and (seq_id + 1, "N") in lookup:
            i = lookup[(seq_id, "C")]
            j = lookup[(seq_id + 1, "N")]
            add_edge(i, j, True)
            add_edge(j, i, True)

    # Spatial graph. Fixed connectivity is constructed from the mutant
    # initialization; distances are recomputed after every coordinate update.
    tree = cKDTree(positions)
    for i, coord in enumerate(positions):
        neighbors = tree.query_ball_point(coord, r=float(config.radius))
        neighbors = [j for j in neighbors if j != i]
        neighbors.sort(key=lambda j: float(np.linalg.norm(positions[j] - coord)))
        for j in neighbors[: int(config.max_neighbors)]:
            add_edge(i, j, False)

    if not edges:
        raise ValueError("Graph contains no edges")

    edge_pairs = sorted(edges)
    edge_index = np.asarray(edge_pairs, dtype=np.int64).T
    edge_features: list[list[float]] = []
    for i, j in edge_pairs:
        seq_i = int(seq_ids[i])
        seq_j = int(seq_ids[j])
        edge_features.append(
            [
                float(edges[(i, j)]),
                float(seq_i == seq_j),
                1.0 / (1.0 + abs(seq_i - seq_j)),
                float(seq_i == mutation_position and seq_j == mutation_position),
            ]
        )
    return edge_index, np.asarray(edge_features, dtype=np.float32)


def _coordinate_map_from_nodes(pair: PairStructure) -> dict[tuple[int, str], np.ndarray]:
    return {(node.seq_id, node.atom_name): node.pos_init for node in pair.nodes}


def _frame_data(
    pair: PairStructure,
    node_residue_ids: list[int],
) -> dict[str, np.ndarray]:
    init_map = _coordinate_map_from_nodes(pair)
    target_map = pair.target_positions
    n_residues = len(node_residue_ids)

    frame_init = np.tile(np.eye(3, dtype=np.float32), (n_residues, 1, 1))
    frame_target_rel = np.tile(np.eye(3, dtype=np.float32), (n_residues, 1, 1))
    translation_target_local = np.zeros((n_residues, 3), dtype=np.float32)
    frame_mask = np.zeros(n_residues, dtype=bool)
    ca_init = np.zeros((n_residues, 3), dtype=np.float32)

    torsion_target = np.zeros((n_residues, 3, 2), dtype=np.float32)
    torsion_target[..., 1] = 1.0  # zero angle -> (sin=0, cos=1)
    torsion_mask = np.zeros((n_residues, 3), dtype=bool)

    def get_frame(coord_map: dict[tuple[int, str], np.ndarray], seq_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        keys = [(seq_id, "N"), (seq_id, "CA"), (seq_id, "C")]
        if not all(key in coord_map for key in keys):
            return None
        n, ca, c = [coord_map[key] for key in keys]
        try:
            return residue_frame_np(n, ca, c), np.asarray(ca, dtype=float)
        except ValueError:
            return None

    def torsions(coord_map: dict[tuple[int, str], np.ndarray], seq_id: int) -> list[float | None]:
        values: list[float | None] = [None, None, None]  # phi, psi, omega
        phi_keys = [
            (seq_id - 1, "C"), (seq_id, "N"), (seq_id, "CA"), (seq_id, "C")
        ]
        psi_keys = [
            (seq_id, "N"), (seq_id, "CA"), (seq_id, "C"), (seq_id + 1, "N")
        ]
        omega_keys = [
            (seq_id - 1, "CA"), (seq_id - 1, "C"), (seq_id, "N"), (seq_id, "CA")
        ]
        for index, keys in enumerate((phi_keys, psi_keys, omega_keys)):
            if all(key in coord_map for key in keys):
                try:
                    values[index] = dihedral_np(*[coord_map[key] for key in keys])
                except ValueError:
                    values[index] = None
        return values

    for ridx, seq_id in enumerate(node_residue_ids):
        init_frame = get_frame(init_map, seq_id)
        target_frame = get_frame(target_map, seq_id)
        if init_frame is not None:
            r0, ca0 = init_frame
            frame_init[ridx] = r0.astype(np.float32)
            ca_init[ridx] = ca0.astype(np.float32)
        if init_frame is not None and target_frame is not None:
            r0, ca0 = init_frame
            rt, cat = target_frame
            frame_target_rel[ridx] = (r0.T @ rt).astype(np.float32)
            translation_target_local[ridx] = (r0.T @ (cat - ca0)).astype(np.float32)
            frame_mask[ridx] = True

        init_torsions = torsions(init_map, seq_id)
        target_torsions = torsions(target_map, seq_id)
        for torsion_index, (initial, target) in enumerate(
            zip(init_torsions, target_torsions, strict=True)
        ):
            if initial is None or target is None:
                continue
            delta = wrap_angle(target - initial)
            torsion_target[ridx, torsion_index] = [np.sin(delta), np.cos(delta)]
            torsion_mask[ridx, torsion_index] = True

    return {
        "residue_frame_init": frame_init,
        "residue_frame_target_rel": frame_target_rel,
        "residue_translation_target_local": translation_target_local,
        "residue_frame_mask": frame_mask,
        "residue_ca_init": ca_init,
        "residue_torsion_target": torsion_target,
        "residue_torsion_mask": torsion_mask,
    }


def build_graph_sample(
    row: dict[str, Any],
    structures_dir: Path,
    config: GraphBuildConfig,
) -> dict[str, Any]:
    pair = build_pair_structure(row=row, structures_dir=structures_dir)
    target_sequence = "".join(str(row["target_sequence"]).split()).upper()

    positions_init = np.asarray([node.pos_init for node in pair.nodes], dtype=np.float32)
    positions_target, target_mask = _target_coordinate_arrays(pair)
    seq_ids = np.asarray([node.seq_id for node in pair.nodes], dtype=np.int64)
    atom_names = [node.atom_name for node in pair.nodes]

    mutation_ca_index = next(
        i for i, node in enumerate(pair.nodes)
        if node.is_mutation_site and node.atom_name == "CA"
    )
    mutation_center = positions_init[mutation_ca_index]
    distance_to_mutation = np.linalg.norm(positions_init - mutation_center, axis=1).astype(np.float32)

    residue_ids = sorted(set(int(x) for x in seq_ids))
    residue_to_index = {seq_id: i for i, seq_id in enumerate(residue_ids)}
    node_residue = np.asarray([residue_to_index[int(x)] for x in seq_ids], dtype=np.int64)

    edge_index, edge_static = _build_edges(
        positions=positions_init,
        seq_ids=seq_ids,
        atom_names=atom_names,
        target_sequence=target_sequence,
        mutation_position=pair.mutation_position,
        config=config,
    )

    frame = _frame_data(pair=pair, node_residue_ids=residue_ids)
    denominator = max(pair.sequence_length - 1, 1)

    result: dict[str, Any] = {
        "schema_version": int(config.cache_schema_version),
        "graph_config": asdict(config),
        "pair_id": str(row.get("pair_id", "")),
        "event_id": str(row.get("event_id", "")),
        "split": str(row.get("split", "")),
        "pair_class": str(row.get("pair_class", "")),
        "reference_pdb_id": str(row.get("reference_pdb_id", "")),
        "target_pdb_id": str(row.get("target_pdb_id", "")),
        "mutation": str(row.get("mutation", "")),
        "mutation_index_0": int(row["mutation_index_0"]),
        "target_alignment_ca_rmsd": float(pair.target_alignment_rmsd),
        "pos_init": torch.from_numpy(positions_init),
        "pos_target": torch.from_numpy(positions_target),
        "target_mask": torch.from_numpy(target_mask),
        "atom_type": torch.tensor(
            [ATOM_TO_ID.get(node.atom_name, UNK_ATOM_ID) for node in pair.nodes],
            dtype=torch.long,
        ),
        "element_type": torch.tensor(
            [_element_id(node.element) for node in pair.nodes],
            dtype=torch.long,
        ),
        "res_type_wt": torch.tensor(
            [AA_TO_ID.get(node.wt_aa, UNK_AA_ID) for node in pair.nodes],
            dtype=torch.long,
        ),
        "res_type_mt": torch.tensor(
            [AA_TO_ID.get(node.mt_aa, UNK_AA_ID) for node in pair.nodes],
            dtype=torch.long,
        ),
        "is_mutation_site": torch.tensor(
            [node.is_mutation_site for node in pair.nodes], dtype=torch.bool
        ),
        "is_new_atom": torch.tensor(
            [node.is_new_atom for node in pair.nodes], dtype=torch.bool
        ),
        "is_ca": torch.tensor(
            [node.atom_name == "CA" for node in pair.nodes], dtype=torch.bool
        ),
        "sequence_position_norm": torch.tensor(
            [(node.seq_id - 1) / denominator for node in pair.nodes],
            dtype=torch.float32,
        ).unsqueeze(-1),
        "distance_to_mutation": torch.from_numpy(distance_to_mutation),
        "node_residue": torch.from_numpy(node_residue),
        "edge_index": torch.from_numpy(edge_index),
        "edge_static": torch.from_numpy(edge_static),
        "residue_seq_id": torch.tensor(residue_ids, dtype=torch.long),
    }
    for key, value in frame.items():
        result[key] = torch.from_numpy(value)
    return result
