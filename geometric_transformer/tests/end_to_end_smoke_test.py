#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data import GraphBatch, PairGraphDataset, collate_graph_samples
from common.geometry import kabsch_mobile_to_reference
from common.graph import GraphBuildConfig
from common.losses import compute_total_loss
from common.templates import load_residue_templates
from network.config import ExperimentConfig
from network.model import GeometricTransformer


def aligned_residue_coords(aa: str, anchors: dict[str, np.ndarray] | None = None):
    template = load_residue_templates()[aa]
    coords = {
        key: np.asarray(value, dtype=float)
        for key, value in template["conformers"][0].items()
        if key != "OXT"
    }
    if anchors is not None:
        names = ["N", "CA", "C"]
        mobile = np.asarray([coords[name] for name in names])
        reference = np.asarray([anchors[name] for name in names])
        rotation, translation, _ = kabsch_mobile_to_reference(mobile, reference)
        coords = {name: xyz @ rotation + translation for name, xyz in coords.items()}
    return coords, template["elements"]


def write_cif(path: Path, sequence: str, mutation_shift: bool) -> None:
    rows = []
    atom_id = 1
    reference_backbones: dict[int, dict[str, np.ndarray]] = {}
    for seq_id, aa in enumerate(sequence, start=1):
        coords, elements = aligned_residue_coords(aa)
        shift = np.asarray([4.0 * (seq_id - 1), 0.2 * seq_id, 0.0])
        coords = {name: xyz + shift for name, xyz in coords.items()}
        if seq_id == 3 and aa == "A":
            reference_backbones[seq_id] = {name: coords[name] for name in ("N", "CA", "C")}
        if seq_id == 3 and aa == "W":
            # Put the mutant on the same backbone as the corresponding ALA
            # geometry used in the reference synthetic structure.
            ala_coords, _ = aligned_residue_coords("A")
            ala_coords = {name: xyz + shift for name, xyz in ala_coords.items()}
            coords, elements = aligned_residue_coords(
                "W", {name: ala_coords[name] for name in ("N", "CA", "C")}
            )
            if mutation_shift:
                for name in coords:
                    if name not in {"N", "CA", "C", "O"}:
                        coords[name] = coords[name] + np.asarray([0.15, -0.05, 0.10])
        for atom_name, xyz in coords.items():
            rows.append(
                [
                    "ATOM", str(atom_id), elements.get(atom_name, atom_name[0]), atom_name,
                    ".", aa, "A", "1", str(seq_id), "?", f"{xyz[0]:.4f}",
                    f"{xyz[1]:.4f}", f"{xyz[2]:.4f}", "1.00", "20.0", "1",
                ]
            )
            atom_id += 1
    tags = [
        "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
        "_atom_site.label_atom_id", "_atom_site.label_alt_id",
        "_atom_site.label_comp_id", "_atom_site.label_asym_id",
        "_atom_site.label_entity_id", "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x",
        "_atom_site.Cartn_y", "_atom_site.Cartn_z", "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv", "_atom_site.pdbx_PDB_model_num",
    ]
    lines = ["data_test", "#", "loop_", *tags]
    lines.extend(" ".join(row) for row in rows)
    lines.append("#")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        structures = temp_path / "structures"
        structures.mkdir()
        write_cif(structures / "ref1.cif", "AAAAA", mutation_shift=False)
        write_cif(structures / "mut1.cif", "AAWAA", mutation_shift=True)
        manifest = pd.DataFrame(
            [
                {
                    "pair_id": "synthetic_train",
                    "event_id": "event1",
                    "split_cluster_id": "cluster1",
                    "split": "train",
                    "pair_class": "canonical_wt",
                    "reference_pdb_id": "REF1",
                    "reference_entity_id": "1",
                    "reference_chain_label_asym_id": "A",
                    "reference_sequence": "AAAAA",
                    "target_pdb_id": "MUT1",
                    "target_entity_id": "1",
                    "target_chain_label_asym_id": "A",
                    "target_sequence": "AAWAA",
                    "mutation_index_0": 2,
                    "mutation": "A3W",
                },
                {
                    "pair_id": "synthetic_valid",
                    "event_id": "event2",
                    "split_cluster_id": "cluster2",
                    "split": "validation",
                    "pair_class": "canonical_wt",
                    "reference_pdb_id": "REF1",
                    "reference_entity_id": "1",
                    "reference_chain_label_asym_id": "A",
                    "reference_sequence": "AAAAA",
                    "target_pdb_id": "MUT1",
                    "target_entity_id": "1",
                    "target_chain_label_asym_id": "A",
                    "target_sequence": "AAWAA",
                    "mutation_index_0": 2,
                    "mutation": "A3W",
                },
            ]
        )
        manifest_path = temp_path / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        cache = temp_path / "cache"

        full_train = PairGraphDataset(
            manifest_path, structures, cache, "train", GraphBuildConfig()
        )
        sample = full_train[0]
        assert int(sample["is_new_atom"].sum()) > 0
        assert int(sample["target_mask"].sum()) > 0
        full_batch = collate_graph_samples([sample])

        local_dataset = PairGraphDataset(
            manifest_path,
            structures,
            cache,
            "train",
            GraphBuildConfig(),
            local_radius=12.0,
        )
        local_batch = collate_graph_samples([local_dataset[0]])

        experiment = ExperimentConfig()
        experiments = [
            (
                "geometric_transformer",
                GeometricTransformer(experiment.model),
                experiment.loss,
                local_batch,
            ),
        ]
        for name, model, weights, batch in experiments:
            model.train()
            outputs = model(batch)
            loss, terms = compute_total_loss(outputs, batch, weights)
            assert torch.isfinite(loss), (name, terms)
            loss.backward()
            grad_norm = sum(
                float(parameter.grad.norm())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            assert grad_norm > 0.0, name

            # Numerical E(3)-equivariance check under a rigid rotation/translation.
            rng = np.random.default_rng(17)
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(rotation) < 0:
                rotation[:, -1] *= -1
            rotation_t = torch.tensor(rotation, dtype=torch.float32)
            translation_t = torch.tensor([1.2, -0.7, 2.1], dtype=torch.float32)
            transformed_data = {
                key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
                for key, value in batch.data.items()
            }
            transformed_data["pos_init"] = batch.pos_init @ rotation_t + translation_t
            transformed_data["pos_target"] = batch.pos_target @ rotation_t + translation_t
            transformed_data["residue_ca_init"] = (
                batch.residue_ca_init @ rotation_t + translation_t
            )
            transformed_data["residue_frame_init"] = (
                rotation_t.T.unsqueeze(0) @ batch.residue_frame_init
            )
            transformed_batch = GraphBatch(transformed_data)
            model.eval()
            with torch.no_grad():
                original_output = model(batch)["pos"]
                transformed_output = model(transformed_batch)["pos"]
            expected_output = original_output @ rotation_t + translation_t
            equivariance_error = float(
                torch.max(torch.abs(transformed_output - expected_output))
            )
            assert equivariance_error < 1e-4, (name, equivariance_error)

            print(
                json.dumps(
                    {
                        "model": name,
                        "nodes": int(batch.pos_init.shape[0]),
                        "edges": int(batch.edge_index.shape[1]),
                        "loss": float(loss.detach()),
                        "gradient_norm_sum": grad_norm,
                        "max_equivariance_error": equivariance_error,
                    }
                )
            )

    print("End-to-end mmCIF/graph/model smoke test passed.")


if __name__ == "__main__":
    main()
