from __future__ import annotations

import torch
import torch.nn as nn

from network.config import ModelConfig
from network.layers import (
    EquivariantCoordinateDecoder,
    ProteinMutationNodeEncoder,
    SparseGeometricStructureEmbedding,
    SparseRPETransformerBlock,
)


class GeometricTransformer(nn.Module):
    """Qin-style geometric attention plus an equivariant coordinate decoder.

    The encoder is invariant to global E(3) transformations because its
    positional information is constructed only from pairwise distances and
    triplet-wise angles. The decoder predicts scalar weights on relative atom
    vectors, making the final coordinate output E(3)-equivariant.

    This is an adaptation for mutation-conditioned coordinate refinement, not a
    copy of the original point-cloud registration network: registration
    cross-attention and correspondence matching are intentionally omitted.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.node_encoder = ProteinMutationNodeEncoder(config.hidden_dim)
        self.geometry_embedding = SparseGeometricStructureEmbedding(
            hidden_dim=config.hidden_dim,
            sigma_distance=config.sigma_distance,
            sigma_angle_degrees=config.sigma_angle_degrees,
            angle_k=config.angle_k,
            angle_reduction=config.angle_reduction,
            dropout=config.dropout,
        )
        self.transformer = nn.ModuleList(
            [
                SparseRPETransformerBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in range(config.transformer_layers)
            ]
        )
        self.coordinate_decoder = EquivariantCoordinateDecoder(
            hidden_dim=config.hidden_dim,
            num_layers=config.refinement_layers,
            dropout=config.dropout,
            distance_cutoff=config.distance_cutoff,
            num_rbf=config.distance_rbf,
            coordinate_scale=config.coordinate_scale,
            movement_gate_bias=config.movement_gate_bias,
        )

    def forward(self, batch) -> dict[str, torch.Tensor]:
        initial_coordinates = batch.pos_init
        node_state = self.node_encoder(batch)

        # Faithful to Qin et al., the distance/angle structure embedding is built
        # once from the input point cloud and reused throughout self-attention.
        geometry = self.geometry_embedding(
            initial_coordinates,
            batch.edge_index,
            batch.edge_static,
            batch.graph_ptr,
        )

        for block in self.transformer:
            node_state, _ = block(
                node_state,
                geometry,
                batch.edge_index,
            )

        node_state, coordinates, movement_gate = self.coordinate_decoder(
            node_state,
            initial_coordinates,
            geometry,
            batch.edge_index,
            batch.edge_static,
        )

        return {
            "pos": coordinates,
            "delta": coordinates - initial_coordinates,
            "node_state": node_state,
            "movement_gate": movement_gate,
        }
