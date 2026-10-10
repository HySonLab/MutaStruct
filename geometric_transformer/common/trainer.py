from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data import GraphBatch, collate_graph_samples, seed_worker
from .losses import LossWeights, compute_total_loss
from .metrics import batch_response_metrics_per_graph, batch_structure_metrics


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 1
    gradient_accumulation: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 15
    num_workers: int = 4
    seed: int = 42
    amp: bool = True
    device: str = "auto"
    log_every: int = 25
    local_metric_radius: float = 8.0
    monitor: str = "loss_total"

    # Fine-tuning sampler. Defaults preserve old behavior.
    response_balanced_sampling: bool = False
    response_bins: int = 5
    response_sampling_radius: float = 8.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(
        requested
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False"
        )

    return device


def _sample_native_local_ca_response(
    sample,
    radius: float,
) -> float:
    """WT-frame native local CA displacement RMSD used only for sampling."""
    target_mask = (
        sample["target_mask"]
        .detach()
        .cpu()
        .numpy()
        .astype(bool)
    )
    is_ca = (
        sample["is_ca"]
        .detach()
        .cpu()
        .numpy()
        .astype(bool)
    )
    distance = (
        sample["distance_to_mutation"]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    initial = (
        sample["pos_init"]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    target = (
        sample["pos_target"]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    mask = (
        target_mask
        & is_ca
        & (
            distance
            <= float(radius)
        )
    )

    if not np.any(mask):
        return 0.0

    delta = (
        target[mask]
        - initial[mask]
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    delta * delta,
                    axis=-1,
                )
            )
        )
    )


def make_response_balanced_sampler(
    dataset,
    bins: int,
    radius: float,
    seed: int,
) -> tuple[WeightedRandomSampler, dict[str, Any]]:
    """Balance training across mutation events and native-response quantiles.

    Uniform pair sampling has two undesirable effects for MutaStruct:
      1. repeated coordinate observations give some mutation events more weight;
      2. the many near-isostructural examples dominate rarer high-response cases.

    This sampler:
      * computes one median native local-CA response score per mutation event,
      * bins events by response quantiles,
      * gives every response bin equal expected mass,
      * gives every event inside a bin equal expected mass,
      * divides an event's mass across its repeated coordinate-pair rows.
    """
    requested_bins = max(
        int(bins),
        1,
    )

    records = []
    event_scores = defaultdict(
        list
    )
    event_pair_count = Counter()

    print(
        "Scanning cached training graphs "
        "to build response-balanced sampler..."
    )

    for index in tqdm(
        range(
            len(dataset)
        ),
        desc="response sampler",
        leave=False,
    ):
        sample = dataset[
            index
        ]

        pair_id = str(
            sample.get(
                "pair_id",
                index,
            )
        )

        event_id = str(
            sample.get(
                "event_id",
                "",
            )
            or pair_id
        )

        score = (
            _sample_native_local_ca_response(
                sample,
                radius=radius,
            )
        )

        records.append(
            (
                event_id,
                score,
            )
        )
        event_scores[
            event_id
        ].append(
            score
        )
        event_pair_count[
            event_id
        ] += 1

    event_response = {
        event_id: float(
            np.median(
                scores
            )
        )
        for event_id, scores
        in event_scores.items()
    }

    values = np.asarray(
        list(
            event_response.values()
        ),
        dtype=float,
    )

    if len(values) == 0:
        raise ValueError(
            "Cannot build response-balanced sampler for an empty dataset"
        )

    if requested_bins == 1:
        internal_edges = np.asarray(
            [],
            dtype=float,
        )
    else:
        quantiles = np.linspace(
            0.0,
            1.0,
            requested_bins + 1,
        )

        raw_edges = np.quantile(
            values,
            quantiles,
        )

        # Internal boundaries only. Repeated quantile values are collapsed.
        internal_edges = np.unique(
            raw_edges[
                1:-1
            ]
        )

    event_bin = {
        event_id: int(
            np.searchsorted(
                internal_edges,
                score,
                side="right",
            )
        )
        for event_id, score
        in event_response.items()
    }

    bin_event_count = Counter(
        event_bin.values()
    )

    weights = []

    for event_id, _ in records:
        bin_id = event_bin[
            event_id
        ]

        weight = 1.0 / (
            float(
                bin_event_count[
                    bin_id
                ]
            )
            * float(
                event_pair_count[
                    event_id
                ]
            )
        )

        weights.append(
            weight
        )

    weights_tensor = torch.tensor(
        weights,
        dtype=torch.double,
    )

    generator = torch.Generator()
    generator.manual_seed(
        int(seed)
    )

    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )

    metadata = {
        "requested_bins": requested_bins,
        "effective_bins": len(
            bin_event_count
        ),
        "events": len(
            event_response
        ),
        "pairs": len(
            dataset
        ),
        "radius": float(
            radius
        ),
        "response_min": float(
            np.min(values)
        ),
        "response_median": float(
            np.median(values)
        ),
        "response_max": float(
            np.max(values)
        ),
        "bin_event_counts": {
            str(k): int(v)
            for k, v
            in sorted(
                bin_event_count.items()
            )
        },
        "internal_edges": [
            float(x)
            for x in internal_edges
        ],
    }

    return sampler, metadata


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    sampler=None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(
            batch_size
        ),
        "shuffle": (
            bool(shuffle)
            if sampler is None
            else False
        ),
        "sampler": sampler,
        "num_workers": int(
            workers
        ),
        "collate_fn": (
            collate_graph_samples
        ),
        "pin_memory": (
            torch.cuda.is_available()
        ),
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": bool(
            workers > 0
        ),
    }

    if workers > 0:
        kwargs[
            "prefetch_factor"
        ] = 2

    return DataLoader(
        **kwargs
    )


def _autocast_context(
    device: torch.device,
    enabled: bool,
):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=(
            enabled
            and device.type == "cuda"
        ),
    )


def _make_scaler(
    enabled: bool,
):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except (
        AttributeError,
        TypeError,
    ):
        return torch.cuda.amp.GradScaler(
            enabled=enabled
        )


def _accumulate(
    accumulator: dict[str, float],
    values: dict[str, float],
    weight: int,
) -> None:
    accumulator[
        "_weight"
    ] = accumulator.get(
        "_weight",
        0.0,
    ) + float(weight)

    for key, value in values.items():
        if math.isfinite(
            float(value)
        ):
            accumulator[
                key
            ] = accumulator.get(
                key,
                0.0,
            ) + float(value) * weight


def _finalize(
    accumulator: dict[str, float],
) -> dict[str, float]:
    weight = accumulator.get(
        "_weight",
        0.0,
    )

    if weight <= 0:
        return {}

    return {
        key: value / weight
        for key, value
        in accumulator.items()
        if key != "_weight"
    }


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    loss_weights: LossWeights,
    config: TrainingConfig,
    epoch: int,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(
        set_to_none=True
    )

    accumulator: dict[
        str,
        float,
    ] = {}

    progress = tqdm(
        loader,
        desc=f"train {epoch}",
        leave=False,
    )

    for step, batch in enumerate(
        progress,
        start=1,
    ):
        batch = batch.to(
            device,
            non_blocking=True,
        )

        with _autocast_context(
            device,
            config.amp,
        ):
            outputs = model(
                batch
            )

            loss, terms = (
                compute_total_loss(
                    outputs,
                    batch,
                    loss_weights,
                )
            )

            scaled_loss = (
                loss
                / max(
                    int(
                        config.gradient_accumulation
                    ),
                    1,
                )
            )

        if not torch.isfinite(
            loss
        ):
            raise FloatingPointError(
                f"Non-finite training loss at epoch={epoch}, "
                f"step={step}, pairs={batch.pair_id}"
            )

        scaler.scale(
            scaled_loss
        ).backward()

        should_step = (
            step
            % max(
                int(
                    config.gradient_accumulation
                ),
                1,
            )
            == 0
            or step
            == len(loader)
        )

        if should_step:
            scaler.unscale_(
                optimizer
            )

            clip_grad_norm_(
                model.parameters(),
                float(
                    config.grad_clip
                ),
            )

            scaler.step(
                optimizer
            )
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        values = {
            f"loss_{key}": float(
                value.detach().cpu()
            )
            for key, value
            in terms.items()
        }

        _accumulate(
            accumulator,
            values,
            batch.num_graphs,
        )

        if (
            step
            % max(
                int(
                    config.log_every
                ),
                1,
            )
            == 0
        ):
            progress.set_postfix(
                loss=(
                    f"{float(loss.detach().cpu()):.4f}"
                )
            )

    return _finalize(
        accumulator
    )


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_weights: LossWeights,
    config: TrainingConfig,
    epoch: int,
) -> dict[str, float]:
    model.eval()

    accumulator: dict[
        str,
        float,
    ] = {}

    # Response-sensitive checkpoint metrics are additionally aggregated by
    # mutation event so repeated coordinate observations do not receive extra
    # validation weight.
    event_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    progress = tqdm(
        loader,
        desc=f"valid {epoch}",
        leave=False,
    )

    for batch in progress:
        batch = batch.to(
            device,
            non_blocking=True,
        )

        with _autocast_context(
            device,
            config.amp,
        ):
            outputs = model(
                batch
            )

            loss, terms = (
                compute_total_loss(
                    outputs,
                    batch,
                    loss_weights,
                )
            )

        if not torch.isfinite(
            loss
        ):
            raise FloatingPointError(
                f"Non-finite validation loss for pairs={batch.pair_id}"
            )

        values = {
            f"loss_{key}": float(
                value.detach().cpu()
            )
            for key, value
            in terms.items()
        }

        values.update(
            batch_structure_metrics(
                outputs[
                    "pos"
                ],
                batch,
                local_radius=(
                    config.local_metric_radius
                ),
            )
        )

        per_graph_response = batch_response_metrics_per_graph(
            outputs["pos"],
            batch,
            local_radius=config.local_metric_radius,
        )

        for graph_index, graph_metrics in enumerate(per_graph_response):
            event_id = str(
                batch.event_id[graph_index]
                or batch.pair_id[graph_index]
            )
            for key, value in graph_metrics.items():
                if math.isfinite(float(value)):
                    event_values[event_id][key].append(float(value))

        _accumulate(
            accumulator,
            values,
            batch.num_graphs,
        )

    final = _finalize(
        accumulator
    )

    # Override response-sensitive validation metrics with event-macro values,
    # matching the benchmark's primary aggregation.  Loss terms and legacy
    # structure metrics remain pair-macro.
    response_keys = [
        "local_ca_rmsd",
        "mr_rmsd_8a",
        "copy_wt_mr_rmsd_8a",
        "mr_excess_8a",
        "mutation_sidechain_rmsd",
    ]

    for key in response_keys:
        per_event = []
        for metrics_by_key in event_values.values():
            values_for_event = metrics_by_key.get(key, [])
            if values_for_event:
                per_event.append(float(np.mean(values_for_event)))
        if per_event:
            final[key] = float(np.mean(per_event))

    return final


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict[str, float],
    configuration: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        {
            "epoch": int(
                epoch
            ),
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
            "metrics": metrics,
            "configuration": (
                configuration
            ),
        },
        temporary,
    )

    os.replace(
        temporary,
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
    | None = None,
    scheduler=None,
) -> tuple[int, dict[str, float]]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict"
        in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    if (
        scheduler is not None
        and "scheduler_state_dict"
        in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

    return (
        int(
            checkpoint.get(
                "epoch",
                0,
            )
        ),
        dict(
            checkpoint.get(
                "metrics",
                {},
            )
        ),
    )


def fit(
    model_name: str,
    model: torch.nn.Module,
    train_dataset,
    validation_dataset,
    loss_weights: LossWeights,
    training_config: TrainingConfig,
    run_dir: Path,
    full_configuration: dict[str, Any],
    resume: Path | None = None,
    init_checkpoint: Path | None = None,
) -> None:
    """Train or fine-tune a model.

    resume:
      Restore model + optimizer + scheduler and continue epoch numbering.

    init_checkpoint:
      Load MODEL WEIGHTS ONLY, then create a fresh optimizer/scheduler and start
      a new run at epoch 1.  This is the recommended mode for response-focused
      fine-tuning from the previous reference refinement best.pt.
    """
    if (
        resume is not None
        and init_checkpoint is not None
    ):
        raise ValueError(
            "Use either resume or init_checkpoint, not both."
        )

    set_seed(
        training_config.seed
    )

    device = resolve_device(
        training_config.device
    )

    run_dir = Path(
        run_dir
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_sampler = None
    sampler_metadata = None

    if (
        training_config.response_balanced_sampling
    ):
        (
            train_sampler,
            sampler_metadata,
        ) = make_response_balanced_sampler(
            train_dataset,
            bins=(
                training_config.response_bins
            ),
            radius=(
                training_config.response_sampling_radius
            ),
            seed=(
                training_config.seed
            ),
        )

        print(
            "Response-balanced sampler:"
        )
        print(
            json.dumps(
                sampler_metadata,
                indent=2,
            )
        )

    train_loader = make_loader(
        train_dataset,
        batch_size=(
            training_config.batch_size
        ),
        shuffle=True,
        workers=(
            training_config.num_workers
        ),
        seed=(
            training_config.seed
        ),
        sampler=train_sampler,
    )

    validation_loader = make_loader(
        validation_dataset,
        batch_size=(
            training_config.batch_size
        ),
        shuffle=False,
        workers=(
            training_config.num_workers
        ),
        seed=(
            training_config.seed
            + 1
        ),
    )

    model = model.to(
        device
    )

    if init_checkpoint is not None:
        loaded_epoch, loaded_metrics = (
            load_checkpoint(
                Path(
                    init_checkpoint
                ),
                model,
                optimizer=None,
                scheduler=None,
            )
        )

        print(
            "Initialized model weights from:"
        )
        print(
            f"  {Path(init_checkpoint)}"
        )
        print(
            f"Source checkpoint epoch: {loaded_epoch}"
        )
        if loaded_metrics:
            print(
                "Source checkpoint metrics:"
            )
            print(
                json.dumps(
                    loaded_metrics,
                    indent=2,
                )
            )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=(
            training_config.learning_rate
        ),
        weight_decay=(
            training_config.weight_decay
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(
                training_config.epochs,
                1,
            ),
        )
    )

    scaler = _make_scaler(
        enabled=(
            training_config.amp
            and device.type == "cuda"
        )
    )

    start_epoch = 1

    if resume is not None:
        resumed_epoch, _ = (
            load_checkpoint(
                Path(
                    resume
                ),
                model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        )

        start_epoch = (
            resumed_epoch + 1
        )

    configuration_for_disk = dict(
        full_configuration
    )

    configuration_for_disk[
        "initialization"
    ] = {
        "resume": (
            str(resume)
            if resume is not None
            else None
        ),
        "init_checkpoint": (
            str(init_checkpoint)
            if init_checkpoint is not None
            else None
        ),
    }

    if sampler_metadata is not None:
        configuration_for_disk[
            "response_sampler"
        ] = sampler_metadata

    configuration_path = (
        run_dir
        / "configuration.json"
    )

    configuration_path.write_text(
        json.dumps(
            configuration_for_disk,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print(
        f"Model:             {model_name}"
    )
    print(
        f"Device:            {device}"
    )
    print(
        f"Train pairs:       {len(train_dataset):,}"
    )
    print(
        f"Validation pairs:  {len(validation_dataset):,}"
    )
    print(
        f"Run directory:     {run_dir}"
    )
    print(
        f"Parameters:        "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    print(
        f"Monitor:           {training_config.monitor}"
    )
    print("=" * 80)

    history: list[
        dict[str, Any]
    ] = []

    best_value = float(
        "inf"
    )
    patience_count = 0
    started = time.time()

    for epoch in range(
        start_epoch,
        training_config.epochs + 1,
    ):
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            loss_weights=loss_weights,
            config=training_config,
            epoch=epoch,
        )

        validation_metrics = (
            validate_epoch(
                model=model,
                loader=validation_loader,
                device=device,
                loss_weights=loss_weights,
                config=training_config,
                epoch=epoch,
            )
        )

        scheduler.step()

        record: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": (
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
            ),
            **{
                f"train_{k}": v
                for k, v
                in train_metrics.items()
            },
            **{
                f"val_{k}": v
                for k, v
                in validation_metrics.items()
            },
        }

        history.append(
            record
        )

        history_path = (
            run_dir
            / "history.csv"
        )

        fieldnames = sorted(
            {
                key
                for row in history
                for key in row
            }
        )

        with history_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(
                history
            )

        monitor = (
            validation_metrics.get(
                training_config.monitor
            )
        )

        if (
            monitor is None
            or not math.isfinite(
                monitor
            )
        ):
            monitor = (
                validation_metrics.get(
                    "loss_total",
                    float("inf"),
                )
            )

        save_checkpoint(
            run_dir
            / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            validation_metrics,
            configuration_for_disk,
        )

        improved = (
            monitor
            < best_value
        )

        if improved:
            best_value = monitor
            patience_count = 0

            save_checkpoint(
                run_dir
                / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                validation_metrics,
                configuration_for_disk,
            )
        else:
            patience_count += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_metrics.get('loss_total', float('nan')):.4f} | "
            f"val={validation_metrics.get('loss_total', float('nan')):.4f} | "
            f"val_CA={validation_metrics.get('ca_rmsd', float('nan')):.4f} A | "
            f"val_MR8={validation_metrics.get('mr_rmsd_8a', float('nan')):.4f} A | "
            f"CopyWT_MR8={validation_metrics.get('copy_wt_mr_rmsd_8a', float('nan')):.4f} A | "
            f"MR_excess={validation_metrics.get('mr_excess_8a', float('nan')):+.4f} A | "
            f"SC={validation_metrics.get('mutation_sidechain_rmsd', float('nan')):.4f} A | "
            f"best={best_value:.4f} | "
            f"patience={patience_count}/{training_config.patience}"
        )

        if (
            patience_count
            >= training_config.patience
        ):
            print(
                "Early stopping triggered."
            )
            break

    summary = {
        "model": model_name,
        "monitor": (
            training_config.monitor
        ),
        "best_monitor_value": (
            best_value
        ),
        "elapsed_seconds": (
            time.time()
            - started
        ),
        "epochs_completed": (
            history[-1][
                "epoch"
            ]
            if history
            else 0
        ),
        "best_checkpoint": str(
            (
                run_dir
                / "best.pt"
            ).resolve()
        ),
        "last_checkpoint": str(
            (
                run_dir
                / "last.pt"
            ).resolve()
        ),
    }

    (
        run_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )
