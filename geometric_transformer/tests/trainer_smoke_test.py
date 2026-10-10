#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data import PairGraphDataset
from common.graph import GraphBuildConfig
from common.trainer import TrainingConfig, fit
from network.config import ExperimentConfig, ModelConfig
from network.model import GeometricTransformer
from tests.end_to_end_smoke_test import write_cif


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        structures = root / "structures"
        structures.mkdir()
        write_cif(structures / "ref1.cif", "AAAAA", mutation_shift=False)
        write_cif(structures / "mut1.cif", "AAWAA", mutation_shift=True)

        rows = []
        for index, split in enumerate(["train", "train", "validation"]):
            rows.append(
                {
                    "pair_id": f"pair_{index}",
                    "event_id": f"event_{index}",
                    "split_cluster_id": f"cluster_{index}",
                    "split": split,
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
                }
            )

        manifest = root / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        graph_config = GraphBuildConfig()
        cache = root / "cache"

        train_dataset = PairGraphDataset(
            manifest,
            structures,
            cache,
            "train",
            graph_config,
            local_radius=12.0,
        )
        validation_dataset = PairGraphDataset(
            manifest,
            structures,
            cache,
            "validation",
            graph_config,
            local_radius=12.0,
        )

        experiment = ExperimentConfig()
        experiment.model = ModelConfig(
            hidden_dim=64,
            num_heads=4,
            transformer_layers=1,
            refinement_layers=1,
            dropout=0.0,
            ffn_multiplier=2,
            angle_k=3,
            distance_rbf=12,
        )
        model = GeometricTransformer(experiment.model)
        training = TrainingConfig(
            epochs=1,
            batch_size=1,
            gradient_accumulation=1,
            learning_rate=1e-4,
            weight_decay=1e-4,
            patience=2,
            num_workers=0,
            amp=False,
            device="cpu",
            log_every=1,
            local_metric_radius=8.0,
            monitor="loss_total",
        )

        run_dir = root / "run"
        fit(
            model_name="geometric_transformer_trainer_smoke",
            model=model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            loss_weights=experiment.loss,
            training_config=training,
            run_dir=run_dir,
            full_configuration={
                "model_name": "geometric_transformer_trainer_smoke",
                "model": experiment.model.__dict__,
                "loss": experiment.loss.__dict__,
                "training": training.__dict__,
                "graph": graph_config.__dict__,
            },
        )

        assert (run_dir / "best.pt").exists()
        assert (run_dir / "last.pt").exists()
        assert (run_dir / "history.csv").exists()
        assert (run_dir / "configuration.json").exists()

    print("One-epoch trainer/checkpoint smoke test passed.")


if __name__ == "__main__":
    main()
