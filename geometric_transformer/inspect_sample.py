#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.constants import DEFAULT_CACHE_DIR, DEFAULT_MANIFEST, DEFAULT_STRUCTURES_DIR
from common.data import PairGraphDataset
from common.graph import GraphBuildConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one cached/constructed graph sample.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--structures-dir", type=Path, default=DEFAULT_STRUCTURES_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    dataset = PairGraphDataset(
        manifest_path=args.manifest,
        structures_dir=args.structures_dir,
        cache_dir=args.cache_dir,
        split=args.split,
        graph_config=GraphBuildConfig(),
    )
    sample = dataset[args.index]
    print(f"pair_id:              {sample['pair_id']}")
    print(f"mutation:             {sample['mutation']}")
    print(f"nodes:                {sample['pos_init'].shape[0]}")
    print(f"edges:                {sample['edge_index'].shape[1]}")
    print(f"residues represented: {sample['residue_seq_id'].shape[0]}")
    print(f"target atoms observed:{int(sample['target_mask'].sum())}")
    print(f"new mutant atoms:     {int(sample['is_new_atom'].sum())}")
    print(f"target align CA RMSD: {sample['target_alignment_ca_rmsd']:.4f} A")
    print(f"frame targets:        {int(sample['residue_frame_mask'].sum())}")
    print(f"torsion targets:      {int(sample['residue_torsion_mask'].sum())}")


if __name__ == "__main__":
    main()
