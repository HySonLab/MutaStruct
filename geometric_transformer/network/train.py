#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.cli import common_parser, launch
from common.trainer import set_seed
from network.config import ExperimentConfig
from network.model import GeometricTransformer


def main() -> None:
    parser = common_parser(
        description=(
            "Train the sparse Qin-inspired GeoTransformer refiner on the "
            "MutaStruct Tier-A train/validation split."
        ),
        model_name="geometric_transformer",
        default_batch_size=2,
        default_gradient_accumulation=4,
    )

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--refinement-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--ffn-multiplier", type=int, default=4)

    parser.add_argument("--sigma-distance", type=float, default=0.50)
    parser.add_argument("--sigma-angle-degrees", type=float, default=15.0)
    parser.add_argument("--angle-k", type=int, default=4)
    parser.add_argument(
        "--angle-reduction",
        choices=["max", "mean"],
        default="max",
    )

    parser.add_argument("--distance-cutoff", type=float, default=8.0)
    parser.add_argument("--distance-rbf", type=int, default=24)
    parser.add_argument("--coordinate-scale", type=float, default=0.10)
    parser.add_argument("--movement-gate-bias", type=float, default=-2.0)
    parser.add_argument("--crop-radius", type=float, default=12.0)

    # A more conservative starting learning rate than the previous reference refinement.
    parser.set_defaults(learning_rate=1e-4, local_metric_radius=8.0)

    args = parser.parse_args()

    config = ExperimentConfig()
    config.model.hidden_dim = args.hidden_dim
    config.model.num_heads = args.heads
    config.model.transformer_layers = args.transformer_layers
    config.model.refinement_layers = args.refinement_layers
    config.model.dropout = args.dropout
    config.model.ffn_multiplier = args.ffn_multiplier
    config.model.sigma_distance = args.sigma_distance
    config.model.sigma_angle_degrees = args.sigma_angle_degrees
    config.model.angle_k = args.angle_k
    config.model.angle_reduction = args.angle_reduction
    config.model.distance_cutoff = args.distance_cutoff
    config.model.distance_rbf = args.distance_rbf
    config.model.coordinate_scale = args.coordinate_scale
    config.model.movement_gate_bias = args.movement_gate_bias
    config.model.crop_radius = args.crop_radius

    set_seed(args.seed)
    model = GeometricTransformer(config.model)

    launch(
        model_name="geometric_transformer",
        model=model,
        experiment_config=config,
        args=args,
        local_radius=config.model.crop_radius,
    )


if __name__ == "__main__":
    main()
