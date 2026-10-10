#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.constants import DEFAULT_CACHE_DIR, DEFAULT_MANIFEST, DEFAULT_STRUCTURES_DIR
from common.data import PairGraphDataset
from common.graph import GraphBuildConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute shared MutaStruct mutant-initialization graph caches."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--structures-dir", type=Path, default=DEFAULT_STRUCTURES_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--graph-radius", type=float, default=6.0)
    parser.add_argument("--max-neighbors", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_config = GraphBuildConfig(
        radius=args.graph_radius,
        max_neighbors=args.max_neighbors,
    )
    all_errors: list[dict[str, str]] = []

    for split in args.splits:
        dataset = PairGraphDataset(
            manifest_path=args.manifest,
            structures_dir=args.structures_dir,
            cache_dir=args.cache_dir,
            split=split,
            graph_config=graph_config,
            max_samples=args.max_samples,
        )
        print(f"Preparing {split}: {len(dataset):,} pairs")
        completed = 0
        with ThreadPoolExecutor(max_workers=max(int(args.workers), 1)) as executor:
            future_to_index = {
                executor.submit(dataset._build_or_load, index): index
                for index in range(len(dataset))
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    row = dataset.frame.iloc[index]
                    all_errors.append(
                        {
                            "split": split,
                            "pair_id": str(row.get("pair_id", "")),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                completed += 1
                if completed % 100 == 0 or completed == len(dataset):
                    print(
                        f"  [{completed}/{len(dataset)}] "
                        f"errors={len(all_errors)}",
                        flush=True,
                    )

    if all_errors:
        error_path = args.cache_dir / "cache_build_errors.csv"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        with error_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_errors[0]))
            writer.writeheader()
            writer.writerows(all_errors)
        print(f"Cache completed with {len(all_errors)} errors: {error_path}")
        raise SystemExit(1)
    print("Cache completed without errors.")


if __name__ == "__main__":
    main()
