from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .graph import GraphBuildConfig, build_graph_sample


NODE_KEYS = [
    "pos_init",
    "pos_target",
    "target_mask",
    "atom_type",
    "element_type",
    "res_type_wt",
    "res_type_mt",
    "is_mutation_site",
    "is_new_atom",
    "is_ca",
    "sequence_position_norm",
    "distance_to_mutation",
]
RESIDUE_KEYS = [
    "residue_seq_id",
    "residue_frame_init",
    "residue_frame_target_rel",
    "residue_translation_target_local",
    "residue_frame_mask",
    "residue_ca_init",
    "residue_torsion_target",
    "residue_torsion_mask",
]
META_KEYS = [
    "pair_id",
    "event_id",
    "split",
    "pair_class",
    "reference_pdb_id",
    "target_pdb_id",
    "mutation",
    "mutation_index_0",
    "target_alignment_ca_rmsd",
]


def read_manifest(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        frame = pd.read_parquet(path)
    elif suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported manifest format: {path}")
    return frame


def _safe_pair_filename(pair_id: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    cleaned = "".join(char if char in allowed else "_" for char in pair_id)
    return cleaned or "pair"


def crop_graph_sample(sample: dict[str, Any], radius: float) -> dict[str, Any]:
    node_mask = (sample["distance_to_mutation"] <= float(radius)) | sample[
        "is_mutation_site"
    ]
    selected = torch.where(node_mask)[0]
    if selected.numel() < 4:
        raise ValueError(
            f"Crop radius {radius} selected only {selected.numel()} atoms for {sample['pair_id']}"
        )

    old_to_new = torch.full(
        (sample["pos_init"].shape[0],), -1, dtype=torch.long
    )
    old_to_new[selected] = torch.arange(selected.numel(), dtype=torch.long)

    edge_index = sample["edge_index"]
    edge_keep = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    cropped_edge = old_to_new[edge_index[:, edge_keep]]

    old_residue = sample["node_residue"][selected]
    selected_residues = torch.unique(old_residue, sorted=True)
    residue_old_to_new = torch.full(
        (sample["residue_seq_id"].shape[0],), -1, dtype=torch.long
    )
    residue_old_to_new[selected_residues] = torch.arange(
        selected_residues.numel(), dtype=torch.long
    )

    result: dict[str, Any] = {}
    for key, value in sample.items():
        if key in NODE_KEYS:
            result[key] = value[selected]
        elif key == "edge_index":
            result[key] = cropped_edge
        elif key == "edge_static":
            result[key] = value[edge_keep]
        elif key == "node_residue":
            result[key] = residue_old_to_new[old_residue]
        elif key in RESIDUE_KEYS:
            result[key] = value[selected_residues]
        else:
            result[key] = value

    result["full_num_nodes"] = int(sample["pos_init"].shape[0])
    result["crop_radius"] = float(radius)
    return result


class PairGraphDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        structures_dir: Path,
        cache_dir: Path,
        split: str,
        graph_config: GraphBuildConfig,
        local_radius: float | None = None,
        pair_class: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.structures_dir = Path(structures_dir)
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.graph_config = graph_config
        self.local_radius = local_radius

        frame = read_manifest(self.manifest_path)
        required = [
            "pair_id",
            "split",
            "reference_pdb_id",
            "reference_entity_id",
            "reference_chain_label_asym_id",
            "reference_sequence",
            "target_pdb_id",
            "target_entity_id",
            "target_chain_label_asym_id",
            "target_sequence",
            "mutation_index_0",
        ]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

        frame = frame[frame["split"].astype(str) == split].copy()
        if pair_class and pair_class != "all":
            if "pair_class" not in frame.columns:
                raise ValueError("pair_class filtering requested, but column is absent")
            frame = frame[frame["pair_class"].astype(str) == pair_class].copy()
        frame = frame.sort_values("pair_id", kind="stable").reset_index(drop=True)
        if max_samples is not None:
            frame = frame.head(int(max_samples)).copy()
        if frame.empty:
            raise ValueError(f"No samples selected for split={split!r}")
        self.frame = frame
        self.cache_split_dir = self.cache_dir / split
        self.cache_split_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.frame)

    def cache_path_for_index(self, index: int) -> Path:
        pair_id = str(self.frame.iloc[index]["pair_id"])
        return self.cache_split_dir / f"{_safe_pair_filename(pair_id)}.pt"

    def _build_or_load(self, index: int) -> dict[str, Any]:
        cache_path = self.cache_path_for_index(index)
        if cache_path.exists():
            sample = torch.load(cache_path, map_location="cpu", weights_only=False)
            expected = int(self.graph_config.cache_schema_version)
            if int(sample.get("schema_version", -1)) == expected:
                return sample

        row = self.frame.iloc[index].to_dict()
        sample = build_graph_sample(
            row=row,
            structures_dir=self.structures_dir,
            config=self.graph_config,
        )
        temporary = cache_path.with_suffix(
            f".tmp.{os.getpid()}.{random.randrange(1_000_000)}"
        )
        torch.save(sample, temporary)
        os.replace(temporary, cache_path)
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._build_or_load(index)
        if self.local_radius is not None:
            sample = crop_graph_sample(sample, self.local_radius)
        return sample


class GraphBatch:
    """Pickle-safe wrapper around a collated graph dictionary."""

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "data", data)

    def __getattr__(self, name: str) -> Any:
        # Do not use self.data here.
        # During unpickling `data` may not exist yet.
        data = object.__getattribute__(self, "__dict__").get("data")

        if data is None:
            raise AttributeError(name)

        try:
            return data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __getstate__(self) -> dict[str, Any]:
        return {"data": self.data}

    def __setstate__(self, state: dict[str, Any]) -> None:
        object.__setattr__(self, "data", state["data"])

    @property
    def num_graphs(self) -> int:
        return len(self.data["pair_id"])

    def pin_memory(self) -> "GraphBatch":
        pinned: dict[str, Any] = {}

        for key, value in self.data.items():
            if torch.is_tensor(value):
                pinned[key] = value.pin_memory()
            else:
                pinned[key] = value

        return GraphBatch(pinned)

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ) -> "GraphBatch":
        moved: dict[str, Any] = {}

        for key, value in self.data.items():
            if torch.is_tensor(value):
                moved[key] = value.to(
                    device,
                    non_blocking=non_blocking,
                )
            else:
                moved[key] = value

        return GraphBatch(moved)


def collate_graph_samples(samples: list[dict[str, Any]]) -> GraphBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")

    output: dict[str, Any] = {key: [] for key in NODE_KEYS + RESIDUE_KEYS}
    output.update({key: [] for key in META_KEYS})
    edge_indices: list[torch.Tensor] = []
    edge_static: list[torch.Tensor] = []
    node_residue: list[torch.Tensor] = []
    node_graph: list[torch.Tensor] = []
    residue_graph: list[torch.Tensor] = []
    graph_ptr = [0]

    node_offset = 0
    residue_offset = 0
    for graph_index, sample in enumerate(samples):
        n_nodes = int(sample["pos_init"].shape[0])
        n_residues = int(sample["residue_seq_id"].shape[0])
        for key in NODE_KEYS:
            output[key].append(sample[key])
        for key in RESIDUE_KEYS:
            output[key].append(sample[key])
        for key in META_KEYS:
            output[key].append(sample.get(key))

        edge_indices.append(sample["edge_index"] + node_offset)
        edge_static.append(sample["edge_static"])
        node_residue.append(sample["node_residue"] + residue_offset)
        node_graph.append(torch.full((n_nodes,), graph_index, dtype=torch.long))
        residue_graph.append(torch.full((n_residues,), graph_index, dtype=torch.long))

        node_offset += n_nodes
        residue_offset += n_residues
        graph_ptr.append(node_offset)

    for key in NODE_KEYS + RESIDUE_KEYS:
        output[key] = torch.cat(output[key], dim=0)
    output["edge_index"] = torch.cat(edge_indices, dim=1)
    output["edge_static"] = torch.cat(edge_static, dim=0)
    output["node_residue"] = torch.cat(node_residue, dim=0)
    output["node_graph"] = torch.cat(node_graph, dim=0)
    output["residue_graph"] = torch.cat(residue_graph, dim=0)
    output["graph_ptr"] = torch.tensor(graph_ptr, dtype=torch.long)
    return GraphBatch(output)


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
