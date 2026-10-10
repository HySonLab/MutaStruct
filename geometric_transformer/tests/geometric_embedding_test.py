#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.layers import SparseGeometricStructureEmbedding


def main() -> None:
    torch.manual_seed(19)
    points = torch.randn(7, 3)
    edges = [
        (source, destination)
        for destination in range(len(points))
        for source in range(len(points))
        if source != destination
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    graph_ptr = torch.tensor([0, len(points)], dtype=torch.long)

    sigma_d = 0.5
    sigma_a = 15.0
    k = 3
    module = SparseGeometricStructureEmbedding(
        hidden_dim=32,
        sigma_distance=sigma_d,
        sigma_angle_degrees=sigma_a,
        angle_k=k,
    )
    sparse_d, sparse_a, valid = module.geometry_indices(
        points,
        edge_index,
        graph_ptr,
    )

    # Dense equations from Qin et al.'s reference implementation.
    distance = torch.cdist(points, points)
    dense_d = distance / sigma_d
    knn = distance.topk(k=k + 1, dim=1, largest=False).indices[:, 1:]
    knn_points = points[knn]
    reference = knn_points - points.unsqueeze(1)

    # dense_angle[i,j,k] = angle(x_k-x_i, x_j-x_i)
    anchor = points.unsqueeze(0) - points.unsqueeze(1)
    reference_expanded = reference.unsqueeze(1).expand(-1, len(points), -1, -1)
    anchor_expanded = anchor.unsqueeze(2).expand(-1, -1, k, -1)
    sine = torch.linalg.norm(
        torch.cross(reference_expanded, anchor_expanded, dim=-1),
        dim=-1,
    )
    cosine = torch.sum(reference_expanded * anchor_expanded, dim=-1)
    dense_a = torch.atan2(sine, cosine) * 180.0 / (sigma_a * math.pi)

    source, destination = edge_index
    expected_d = dense_d[destination, source]
    expected_a = dense_a[destination, source]

    assert valid.all()
    d_error = float(torch.max(torch.abs(sparse_d - expected_d)))
    a_error = float(torch.max(torch.abs(sparse_a - expected_a)))
    assert d_error < 1e-6, d_error
    assert a_error < 1e-6, a_error

    print("Sparse Qin geometry-index equivalence test passed")
    print(f"distance-index max error: {d_error:.3e}")
    print(f"angle-index max error:    {a_error:.3e}")


if __name__ == "__main__":
    main()
