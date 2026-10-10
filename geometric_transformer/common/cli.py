from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_STRUCTURES_DIR,
    MODEL_ROOT,
)
from .data import PairGraphDataset
from .graph import GraphBuildConfig
from .trainer import TrainingConfig, fit


def common_parser(
    description: str,
    model_name: str,
    default_batch_size: int,
    default_gradient_accumulation: int,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=DEFAULT_STRUCTURES_DIR,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=(
            MODEL_ROOT.parent
            / "runs"
            / "geometric_transformer"
        ),
    )

    parser.add_argument(
        "--pair-class",
        default="all",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-validation",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch_size,
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=default_gradient_accumulation,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--local-metric-radius",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--monitor",
        choices=[
            "loss_total",
            "ca_rmsd",
            "local_rmsd",
            "local_ca_rmsd",
            "all_atom_rmsd",
            "mr_rmsd_8a",
            "mr_excess_8a",
            "mutation_sidechain_rmsd",
        ],
        default="loss_total",
    )

    # Continue an interrupted run, including optimizer/scheduler state.
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    # Fine-tune from model weights only with a fresh optimizer/scheduler.
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
    )

    # Response-balanced training sampler.
    parser.add_argument(
        "--response-balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--response-bins",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--response-sampling-radius",
        type=float,
        default=8.0,
    )

    # Loss coefficient overrides.  None means use the model's existing config.
    parser.add_argument(
        "--loss-coordinate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--loss-edge-distance",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--loss-clash",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--loss-response-ca",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--loss-mutation-sidechain",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--loss-mr-margin",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--loss-low-response-anchor",
        type=float,
        default=None,
    )

    # Existing coordinate-loss emphasis.
    parser.add_argument(
        "--local-boost",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--local-scale",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--new-atom-boost",
        type=float,
        default=None,
    )

    # Response-loss shape.
    parser.add_argument(
        "--response-radius",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--response-boost",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--response-scale",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--response-max-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--response-beta",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--sidechain-beta",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--mr-radius",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--mr-high-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--mr-low-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--mr-temperature",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--mr-improvement-fraction",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--mr-hinge-softness",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--anchor-beta",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--graph-radius",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=32,
    )

    return parser


def _loss_overrides(
    experiment_loss,
    args: argparse.Namespace,
):
    field_to_arg = {
        "coordinate": "loss_coordinate",
        "edge_distance": "loss_edge_distance",
        "clash": "loss_clash",
        "response_ca": "loss_response_ca",
        "mutation_sidechain": "loss_mutation_sidechain",
        "mr_margin": "loss_mr_margin",
        "low_response_anchor": "loss_low_response_anchor",
        "local_boost": "local_boost",
        "local_scale": "local_scale",
        "new_atom_boost": "new_atom_boost",
        "response_radius": "response_radius",
        "response_boost": "response_boost",
        "response_scale": "response_scale",
        "response_max_weight": "response_max_weight",
        "response_beta": "response_beta",
        "sidechain_beta": "sidechain_beta",
        "mr_radius": "mr_radius",
        "mr_high_threshold": "mr_high_threshold",
        "mr_low_threshold": "mr_low_threshold",
        "mr_temperature": "mr_temperature",
        "mr_improvement_fraction": "mr_improvement_fraction",
        "mr_hinge_softness": "mr_hinge_softness",
        "anchor_beta": "anchor_beta",
    }

    overrides = {}

    for field_name, arg_name in field_to_arg.items():
        value = getattr(
            args,
            arg_name,
        )

        if value is not None:
            overrides[
                field_name
            ] = value

    return (
        replace(
            experiment_loss,
            **overrides,
        ),
        overrides,
    )


def launch(
    model_name: str,
    model,
    experiment_config,
    args: argparse.Namespace,
    local_radius: float | None,
) -> None:
    if (
        args.resume is not None
        and args.init_checkpoint is not None
    ):
        raise ValueError(
            "--resume and --init-checkpoint are mutually exclusive"
        )

    graph_config = GraphBuildConfig(
        radius=args.graph_radius,
        max_neighbors=args.max_neighbors,
    )

    pair_class = (
        None
        if args.pair_class == "all"
        else args.pair_class
    )

    train_dataset = PairGraphDataset(
        manifest_path=args.manifest,
        structures_dir=args.structures_dir,
        cache_dir=args.cache_dir,
        split="train",
        graph_config=graph_config,
        local_radius=local_radius,
        pair_class=pair_class,
        max_samples=args.max_train,
    )

    validation_dataset = (
        PairGraphDataset(
            manifest_path=args.manifest,
            structures_dir=args.structures_dir,
            cache_dir=args.cache_dir,
            split="validation",
            graph_config=graph_config,
            local_radius=local_radius,
            pair_class=pair_class,
            max_samples=args.max_validation,
        )
    )

    training = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        num_workers=args.workers,
        seed=args.seed,
        amp=args.amp,
        device=args.device,
        log_every=args.log_every,
        local_metric_radius=args.local_metric_radius,
        monitor=args.monitor,
        response_balanced_sampling=(
            args.response_balanced_sampling
        ),
        response_bins=args.response_bins,
        response_sampling_radius=(
            args.response_sampling_radius
        ),
    )

    loss_weights, overrides = (
        _loss_overrides(
            experiment_config.loss,
            args,
        )
    )

    configuration: dict[
        str,
        Any,
    ] = {
        "model_name": model_name,
        "model": asdict(
            experiment_config.model
        ),
        "loss": asdict(
            loss_weights
        ),
        "loss_overrides": (
            overrides
        ),
        "training": asdict(
            training
        ),
        "graph": asdict(
            graph_config
        ),
        "paths": {
            "manifest": str(
                args.manifest
            ),
            "structures_dir": str(
                args.structures_dir
            ),
            "cache_dir": str(
                args.cache_dir
            ),
            "run_dir": str(
                args.run_dir
            ),
        },
        "pair_class": (
            args.pair_class
        ),
        "local_radius": (
            local_radius
        ),
    }

    fit(
        model_name=model_name,
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        loss_weights=loss_weights,
        training_config=training,
        run_dir=args.run_dir,
        full_configuration=configuration,
        resume=args.resume,
        init_checkpoint=(
            args.init_checkpoint
        ),
    )
