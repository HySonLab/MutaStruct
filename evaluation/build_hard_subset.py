#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROJECT_DEFAULT = Path(__file__).resolve().parents[1]
MANIFEST_DEFAULT = PROJECT_DEFAULT / 'data/tier_a_pairs_with_splits.parquet'


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {'.parquet', '.pq'}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {'.csv', '.txt'}:
        return pd.read_csv(path)
    raise ValueError(f'Unsupported table format: {path}')


def find_copywt_metrics(project_root: Path, test_pair_ids: set[str]) -> Path:
    candidates = []
    for path in project_root.rglob('pair_metrics.csv'):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        label_col = 'baseline' if 'baseline' in df.columns else ('model' if 'model' in df.columns else None)
        if label_col is None or 'pair_id' not in df.columns or 'local_ca_rmsd_8a' not in df.columns:
            continue
        labels = df[label_col].astype(str)
        mask = labels.isin(['Copy-WT', 'CopyWT'])
        if not mask.any():
            continue
        ids = set(df.loc[mask, 'pair_id'].astype(str))
        overlap = len(ids & test_pair_ids)
        if overlap:
            candidates.append((overlap, path, len(ids)))
    if not candidates:
        raise FileNotFoundError(
            'Could not auto-discover a test Copy-WT pair_metrics.csv. '
            'Pass --copywt-pair-metrics explicitly.'
        )
    candidates.sort(key=lambda x: (x[0], -abs(x[2] - len(test_pair_ids))), reverse=True)
    best = candidates[0]
    if best[0] < len(test_pair_ids):
        print(
            f'WARNING: best Copy-WT source covers only {best[0]}/{len(test_pair_ids)} test pairs: {best[1]}',
            file=sys.stderr,
        )
    return best[1]


def main():
    p = argparse.ArgumentParser(
        description=(
            'Create a hard MutaStruct Tier-A test subset defined at the mutation-event level by '
            'median native WT->MT local C-alpha RMSD within 8 A > threshold.'
        )
    )
    p.add_argument('--project-root', type=Path, default=PROJECT_DEFAULT)
    p.add_argument('--manifest', type=Path, default=MANIFEST_DEFAULT)
    p.add_argument('--copywt-pair-metrics', type=Path, default=None)
    p.add_argument('--threshold', type=float, default=1.0)
    p.add_argument('--output-dir', type=Path, default=PROJECT_DEFAULT / 'results/hard_subset_local_ca_gt1')
    p.add_argument('--expected-events', type=int, default=33)
    p.add_argument('--expected-pairs', type=int, default=75)
    args = p.parse_args()

    manifest = read_table(args.manifest).copy()
    if 'split' not in manifest.columns:
        raise ValueError('Manifest must contain a split column.')
    test = manifest[manifest['split'].astype(str).str.lower().eq('test')].copy()
    if test.empty:
        raise ValueError('No test rows found in the manifest.')
    test['pair_id'] = test['pair_id'].astype(str)
    test['event_id'] = test['event_id'].astype(str)
    test_pair_ids = set(test['pair_id'])

    copywt_path = args.copywt_pair_metrics
    if copywt_path is None:
        copywt_path = find_copywt_metrics(args.project_root, test_pair_ids)
        print(f'Auto-discovered Copy-WT metrics: {copywt_path}')

    pm = pd.read_csv(copywt_path).copy()
    label_col = 'baseline' if 'baseline' in pm.columns else ('model' if 'model' in pm.columns else None)
    if label_col is None:
        raise ValueError('Copy-WT pair metrics must contain model or baseline column.')
    pm = pm[pm[label_col].astype(str).isin(['Copy-WT', 'CopyWT'])].copy()
    pm['pair_id'] = pm['pair_id'].astype(str)
    pm['event_id'] = pm['event_id'].astype(str)
    pm = pm[pm['pair_id'].isin(test_pair_ids)].copy()

    required = {'pair_id', 'event_id', 'local_ca_rmsd_8a'}
    missing = required - set(pm.columns)
    if missing:
        raise ValueError(f'Copy-WT pair metrics missing columns: {sorted(missing)}')

    if pm['pair_id'].duplicated().any():
        dup = pm.loc[pm['pair_id'].duplicated(False), 'pair_id'].unique()[:10]
        raise ValueError(f'Duplicate Copy-WT rows for pair IDs, e.g. {dup.tolist()}')

    merged = test.merge(
        pm[['pair_id', 'event_id', 'local_ca_rmsd_8a'] + ([c for c in ['native_wt_mt_ca_rmsd'] if c in pm.columns])],
        on='pair_id', how='left', suffixes=('', '_copywt')
    )
    if merged['local_ca_rmsd_8a'].isna().any():
        missing_ids = merged.loc[merged['local_ca_rmsd_8a'].isna(), 'pair_id'].head(10).tolist()
        raise ValueError(
            f'Copy-WT local metric missing for {merged.local_ca_rmsd_8a.isna().sum()} test pairs, e.g. {missing_ids}'
        )
    if 'event_id_copywt' in merged.columns:
        mismatch = merged['event_id'].astype(str) != merged['event_id_copywt'].astype(str)
        if mismatch.any():
            raise ValueError('event_id mismatch between manifest and Copy-WT metrics.')

    # The subset is defined at the biological mutation-event level.  For events with
    # repeated coordinate observations, use the median native local structural change.
    event_native = merged.groupby('event_id', as_index=False).agg(
        n_coordinate_pairs=('pair_id', 'count'),
        native_local_ca_rmsd_8a=('local_ca_rmsd_8a', 'median'),
        native_local_ca_rmsd_8a_min=('local_ca_rmsd_8a', 'min'),
        native_local_ca_rmsd_8a_max=('local_ca_rmsd_8a', 'max'),
    )
    if 'native_wt_mt_ca_rmsd' in merged.columns:
        glob = merged.groupby('event_id')['native_wt_mt_ca_rmsd'].median().rename('native_global_ca_rmsd')
        event_native = event_native.merge(glob, on='event_id', how='left')

    hard_events = event_native[event_native['native_local_ca_rmsd_8a'] > float(args.threshold)].copy()
    hard_events = hard_events.sort_values('native_local_ca_rmsd_8a', ascending=False).reset_index(drop=True)
    hard_ids = set(hard_events['event_id'].astype(str))

    hard_manifest = test[test['event_id'].astype(str).isin(hard_ids)].copy()
    hard_manifest = hard_manifest.merge(
        hard_events[['event_id', 'native_local_ca_rmsd_8a', 'n_coordinate_pairs']],
        on='event_id', how='left'
    )
    hard_pairs = merged[merged['event_id'].astype(str).isin(hard_ids)].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hard_events.to_csv(args.output_dir / 'hard_subset_events.csv', index=False)
    hard_pairs.to_csv(args.output_dir / 'hard_subset_pairs.csv', index=False)
    hard_manifest.to_csv(args.output_dir / 'hard_subset_manifest.csv', index=False)
    try:
        hard_manifest.to_parquet(args.output_dir / 'hard_subset_manifest.parquet', index=False)
    except Exception as exc:
        print(f'NOTE: parquet output skipped: {exc}', file=sys.stderr)

    hard_manifest[['pair_id']].drop_duplicates().to_csv(args.output_dir / 'hard_pair_ids.csv', index=False)
    hard_events[['event_id']].to_csv(args.output_dir / 'hard_event_ids.csv', index=False)

    n_events = hard_events['event_id'].nunique()
    n_pairs = hard_manifest['pair_id'].nunique()
    summary = {
        'definition': 'event-level median native WT->MT local C-alpha RMSD within reference-defined 8 A mutation neighborhood > threshold',
        'threshold_angstrom': float(args.threshold),
        'strict_inequality': True,
        'aggregation_for_subset_definition': 'median across coordinate-pair observations within mutation event',
        'test_events_total': int(test['event_id'].nunique()),
        'test_pairs_total': int(test['pair_id'].nunique()),
        'hard_events': int(n_events),
        'hard_pairs': int(n_pairs),
        'hard_event_fraction': float(n_events / test['event_id'].nunique()),
        'hard_pair_fraction': float(n_pairs / test['pair_id'].nunique()),
        'copywt_pair_metrics_source': str(copywt_path),
        'manifest_source': str(args.manifest),
    }
    (args.output_dir / 'hard_subset_metadata.json').write_text(json.dumps(summary, indent=2) + '\n')

    print(json.dumps(summary, indent=2))

    if args.expected_events is not None and n_events != args.expected_events:
        raise SystemExit(f'Expected {args.expected_events} hard events, found {n_events}.')
    if args.expected_pairs is not None and n_pairs != args.expected_pairs:
        raise SystemExit(f'Expected {args.expected_pairs} hard pairs, found {n_pairs}.')


if __name__ == '__main__':
    main()
