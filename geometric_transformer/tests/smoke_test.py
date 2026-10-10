#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data import GraphBatch
from common.losses import compute_total_loss
from network.config import ExperimentConfig, ModelConfig
from network.model import GeometricTransformer


def random_rotation(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    matrix = torch.randn(3, 3, dtype=dtype)
    q, _ = torch.linalg.qr(matrix)
    if torch.det(q) < 0:
        q[:, -1] *= -1
    return q


def fully_connected_edges(sizes: list[int]) -> torch.Tensor:
    edges = []
    offset = 0
    for size in sizes:
        for destination in range(offset, offset + size):
            for source in range(offset, offset + size):
                if source != destination:
                    edges.append((source, destination))
        offset += size
    return torch.tensor(edges, dtype=torch.long).T.contiguous()


def synthetic_batch() -> GraphBatch:
    torch.manual_seed(7)
    sizes = [9, 8]
    num_nodes = sum(sizes)
    graph_ptr = torch.tensor([0, sizes[0], num_nodes], dtype=torch.long)
    node_graph = torch.cat(
        [
            torch.full((size,), graph_index, dtype=torch.long)
            for graph_index, size in enumerate(sizes)
        ]
    )

    # Non-degenerate atom clouds with well-separated graphs.
    x0 = torch.randn(num_nodes, 3)
    x0[sizes[0] :] += torch.tensor([12.0, -4.0, 3.0])
    target = x0.clone()
    target[1] += torch.tensor([0.25, -0.05, 0.10])
    target[3] += torch.tensor([-0.15, 0.10, 0.05])
    target[sizes[0] + 2] += torch.tensor([0.10, 0.20, -0.12])

    edge_index = fully_connected_edges(sizes)
    num_edges = edge_index.shape[1]
    edge_static = torch.zeros(num_edges, 4)
    edge_static[:, 2] = 1.0

    is_ca = torch.zeros(num_nodes, dtype=torch.bool)
    # Three CA atoms in each graph for Kabsch-based optional losses/metrics.
    is_ca[[0, 2, 4, sizes[0], sizes[0] + 2, sizes[0] + 4]] = True

    mutation_site = torch.zeros(num_nodes, dtype=torch.bool)
    mutation_site[[1, 2, sizes[0] + 1, sizes[0] + 2]] = True

    distance = torch.linalg.norm(
        x0
        - torch.stack(
            [x0[2] if g == 0 else x0[sizes[0] + 2] for g in node_graph]
        ),
        dim=-1,
    )

    atom_type = torch.arange(num_nodes, dtype=torch.long) % 10
    element_type = torch.arange(num_nodes, dtype=torch.long) % 4
    res_wt = torch.full((num_nodes,), 0, dtype=torch.long)
    res_mt = res_wt.clone()
    res_mt[mutation_site] = 17

    return GraphBatch(
        {
            "pos_init": x0,
            "pos_target": target,
            "target_mask": torch.ones(num_nodes, dtype=torch.bool),
            "atom_type": atom_type,
            "element_type": element_type,
            "res_type_wt": res_wt,
            "res_type_mt": res_mt,
            "is_mutation_site": mutation_site,
            "is_new_atom": torch.zeros(num_nodes, dtype=torch.bool),
            "is_ca": is_ca,
            "sequence_position_norm": torch.linspace(0, 1, num_nodes).unsqueeze(-1),
            "distance_to_mutation": distance,
            "edge_index": edge_index,
            "edge_static": edge_static,
            "node_graph": node_graph,
            "graph_ptr": graph_ptr,
            "pair_id": ["synthetic_0", "synthetic_1"],
            "event_id": ["event_0", "event_1"],
        }
    )


def transformed_batch(batch: GraphBatch, rotation: torch.Tensor, translation: torch.Tensor) -> GraphBatch:
    data = {}
    for key, value in batch.data.items():
        data[key] = value.clone() if torch.is_tensor(value) else value
    data["pos_init"] = batch.pos_init @ rotation + translation
    data["pos_target"] = batch.pos_target @ rotation + translation
    return GraphBatch(data)


def main() -> None:
    torch.manual_seed(11)
    config = ModelConfig(
        hidden_dim=64,
        num_heads=4,
        transformer_layers=2,
        refinement_layers=2,
        dropout=0.0,
        ffn_multiplier=2,
        angle_k=3,
        distance_rbf=12,
    )
    model = GeometricTransformer(config)
    batch = synthetic_batch()

    model.train()
    output = model(batch)
    assert output["pos"].shape == batch.pos_init.shape
    assert output["node_state"].shape == (batch.pos_init.shape[0], 64)
    assert torch.isfinite(output["pos"]).all()
    assert torch.isfinite(output["movement_gate"]).all()

    experiment = ExperimentConfig()
    loss, terms = compute_total_loss(output, batch, experiment.loss)
    assert torch.isfinite(loss), terms
    loss.backward()

    gradient_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient_norm += float(parameter.grad.detach().norm())
    assert gradient_norm > 0.0

    # Numerical E(3) equivariance test.
    model.eval()
    with torch.no_grad():
        original = model(batch)
        rotation = random_rotation()
        translation = torch.tensor([2.0, -3.5, 1.25])
        moved_batch = transformed_batch(batch, rotation, translation)
        moved = model(moved_batch)

        expected_coordinates = original["pos"] @ rotation + translation
        coordinate_error = torch.max(
            torch.abs(moved["pos"] - expected_coordinates)
        ).item()
        state_error = torch.max(
            torch.abs(moved["node_state"] - original["node_state"])
        ).item()
        gate_error = torch.max(
            torch.abs(moved["movement_gate"] - original["movement_gate"])
        ).item()

    assert coordinate_error < 3e-5, coordinate_error
    assert state_error < 3e-5, state_error
    assert gate_error < 3e-5, gate_error

    parameter_count = sum(p.numel() for p in model.parameters())
    print("Sparse Geometric Transformer smoke test passed")
    print(f"parameters:                {parameter_count:,}")
    print(f"loss:                      {float(loss.detach()):.6f}")
    print(f"backward gradient norm:    {gradient_norm:.6f}")
    print(f"coordinate equivariance:   {coordinate_error:.3e}")
    print(f"feature invariance:        {state_error:.3e}")
    print(f"movement-gate invariance:  {gate_error:.3e}")


if __name__ == "__main__":
    main()
