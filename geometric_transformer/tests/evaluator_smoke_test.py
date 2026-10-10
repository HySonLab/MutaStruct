#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.config import ModelConfig
from network.model import GeometricTransformer
from tests.end_to_end_smoke_test import write_cif


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        structures = root / "structures"
        structures.mkdir()
        write_cif(structures / "ref1.cif", "AAAAA", mutation_shift=False)
        write_cif(structures / "mut1.cif", "AAWAA", mutation_shift=True)

        manifest = root / "manifest.csv"
        pd.DataFrame(
            [
                {
                    "pair_id": "synthetic_test",
                    "event_id": "synthetic_event",
                    "split_cluster_id": "synthetic_cluster",
                    "split": "test",
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
            ]
        ).to_csv(manifest, index=False)

        config = ModelConfig(
            hidden_dim=64,
            num_heads=4,
            transformer_layers=1,
            refinement_layers=1,
            dropout=0.0,
            ffn_multiplier=2,
            angle_k=3,
            distance_rbf=12,
        )
        model = GeometricTransformer(config)
        checkpoint = root / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": 0,
                "metrics": {},
                "configuration": {
                    "model": config.__dict__,
                    "graph": {
                        "radius": 6.0,
                        "max_neighbors": 32,
                        "cache_schema_version": 1,
                    },
                },
            },
            checkpoint,
        )

        output = root / "evaluation"
        command = [
            sys.executable,
            str(ROOT / "evaluate.py"),
            "--manifest",
            str(manifest),
            "--structures-dir",
            str(structures),
            "--cache-dir",
            str(root / "cache"),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--max-cases",
            "1",
            "--device",
            "cpu",
            "--no-amp",
        ]
        subprocess.run(command, check=True)

        assert (output / "pair_metrics.csv").exists()
        assert (output / "event_metrics.csv").exists()
        assert (output / "main_table_event_macro.csv").exists()

    print("Five-metric evaluator smoke test passed.")


if __name__ == "__main__":
    main()
