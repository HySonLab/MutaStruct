from __future__ import annotations

from dataclasses import dataclass, field

from common.losses import LossWeights


@dataclass
class ModelConfig:
    """Sparse Qin-inspired GeoTransformer for local mutant refinement."""

    hidden_dim: int = 128
    num_heads: int = 8
    transformer_layers: int = 4
    refinement_layers: int = 3
    dropout: float = 0.10
    ffn_multiplier: int = 4

    # Qin-style geometric structure embedding.
    sigma_distance: float = 0.50
    sigma_angle_degrees: float = 15.0
    angle_k: int = 4
    angle_reduction: str = "max"

    # Equivariant coordinate decoder.
    distance_cutoff: float = 8.0
    distance_rbf: int = 24
    coordinate_scale: float = 0.10
    movement_gate_bias: float = -2.0

    crop_radius: float = 12.0


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)

    # Keep the original reference refinement loss for the first controlled architecture
    # comparison. Optional response-focused terms in common.losses remain zero.
    loss: LossWeights = field(
        default_factory=lambda: LossWeights(
            coordinate=1.0,
            edge_distance=0.20,
            clash=0.02,
            far_delta=0.0,
            frame_rotation=0.0,
            frame_translation=0.0,
            torsion=0.0,
            response_ca=0.0,
            mutation_sidechain=0.0,
            mr_margin=0.0,
            low_response_anchor=0.0,
            local_boost=3.0,
            local_scale=6.0,
            new_atom_boost=1.0,
        )
    )
