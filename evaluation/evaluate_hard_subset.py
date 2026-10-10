#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd

PROJECT_DEFAULT = Path(__file__).resolve().parents[1]
METRICS = [
    'ca_rmsd',
    'lddt',
    'local_ca_rmsd_8a',
    'mutation_sidechain_rmsd',
    'mutation_response_rmsd_8a',
]
LOWER_IS_BETTER = {m: (m != 'lddt') for m in METRICS}


def normalize_model_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'model' not in out.columns:
        if 'baseline' in out.columns:
            out = out.rename(columns={'baseline': 'model'})
        else:
            raise ValueError('No model or baseline column.')
    out['model'] = out['model'].astype(str)
    out['pair_id'] = out['pair_id'].astype(str)
    out['event_id'] = out['event_id'].astype(str)
    out['model'] = out['model'].replace({'CopyWT': 'Copy-WT', 'AlphaFold3': 'AlphaFold 3', 'Qin-GeoTransformer': 'Geometric Transformer'})
    return out


def load_source(path: Path) -> list[tuple[str, pd.DataFrame]]:
    path = Path(path)
    outputs = []
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.endswith('pair_metrics.csv')]
            for name in names:
                try:
                    df = pd.read_csv(zf.open(name))
                    df = normalize_model_column(df)
                except Exception:
                    continue
                outputs.append((f'{path}::{name}', df))
        return outputs
    if path.name != 'pair_metrics.csv' and path.is_dir():
        for f in path.rglob('pair_metrics.csv'):
            outputs.extend(load_source(f))
        return outputs
    if path.is_file():
        df = normalize_model_column(pd.read_csv(path))
        outputs.append((str(path), df))
        return outputs
    raise FileNotFoundError(path)


def discover_sources(project_root: Path) -> list[Path]:
    return sorted(project_root.rglob('pair_metrics.csv'))


def choose_model_sources(source_tables, hard_pair_ids, hard_event_ids):
    records = []
    per_model = {}
    for source, df in source_tables:
        if not {'pair_id', 'event_id', 'model'}.issubset(df.columns):
            continue
        sub = df[df['pair_id'].isin(hard_pair_ids) & df['event_id'].isin(hard_event_ids)].copy()
        if sub.empty:
            continue
        for model, g in sub.groupby('model', sort=False):
            rec = {
                'model': model,
                'source': source,
                'hard_pairs_covered': int(g['pair_id'].nunique()),
                'hard_events_covered': int(g['event_id'].nunique()),
            }
            records.append(rec)
            per_model.setdefault(model, []).append((rec, g.copy()))

    chosen = []
    selected_records = []
    for model, options in sorted(per_model.items()):
        options.sort(
            key=lambda x: (x[0]['hard_events_covered'], x[0]['hard_pairs_covered'], -len(x[0]['source'])),
            reverse=True,
        )
        best_rec, best_g = options[0]
        best_g['source_file'] = best_rec['source']
        chosen.append(best_g)
        best_rec = dict(best_rec)
        best_rec['selected'] = True
        best_rec['n_alternative_sources'] = len(options) - 1
        selected_records.append(best_rec)
        for rec, _ in options[1:]:
            r = dict(rec)
            r['selected'] = False
            r['n_alternative_sources'] = len(options) - 1
            selected_records.append(r)
    if not chosen:
        raise RuntimeError('No evaluation pair_metrics rows overlap the hard subset.')
    return pd.concat(chosen, ignore_index=True), pd.DataFrame(selected_records), pd.DataFrame(records)


def event_macro(pair_df: pd.DataFrame) -> pd.DataFrame:
    agg = {m: 'mean' for m in METRICS if m in pair_df.columns}
    out = pair_df.groupby(['model', 'event_id'], as_index=False).agg(agg)
    return out


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def make_summary(event_df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for model, g in event_df.groupby('model', sort=False):
        row = {'model': model, 'n_events_any_metric': int(g['event_id'].nunique())}
        for metric in METRICS:
            if metric not in g.columns:
                row[f'{metric}_n'] = 0
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_median'] = np.nan
                row[f'{metric}_ci95_low'] = np.nan
                row[f'{metric}_ci95_high'] = np.nan
                continue
            vals = pd.to_numeric(g[metric], errors='coerce').to_numpy(float)
            vals = vals[np.isfinite(vals)]
            low, high = bootstrap_mean_ci(vals, n_boot, rng)
            row[f'{metric}_n'] = int(len(vals))
            row[f'{metric}_mean'] = float(np.mean(vals)) if len(vals) else np.nan
            row[f'{metric}_median'] = float(np.median(vals)) if len(vals) else np.nan
            row[f'{metric}_ci95_low'] = low
            row[f'{metric}_ci95_high'] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_vs_copywt(event_df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    if 'Copy-WT' not in set(event_df['model']):
        return pd.DataFrame()
    rng = np.random.default_rng(seed + 101)
    copy = event_df[event_df['model'] == 'Copy-WT'].set_index('event_id')
    rows = []
    for model, g in event_df.groupby('model'):
        if model == 'Copy-WT':
            continue
        gg = g.set_index('event_id')
        common_events = sorted(set(copy.index) & set(gg.index))
        for metric in ['ca_rmsd', 'local_ca_rmsd_8a', 'mutation_response_rmsd_8a']:
            if metric not in copy.columns or metric not in gg.columns:
                continue
            a = pd.to_numeric(gg.loc[common_events, metric], errors='coerce').to_numpy(float)
            b = pd.to_numeric(copy.loc[common_events, metric], errors='coerce').to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            if len(a) == 0:
                continue
            delta = a - b  # negative is better for all three metrics here
            idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
            bm = delta[idx].mean(axis=1)
            rows.append({
                'model': model,
                'baseline': 'Copy-WT',
                'metric': metric,
                'n_common_events': int(len(delta)),
                'model_mean': float(a.mean()),
                'copywt_mean': float(b.mean()),
                'model_minus_copywt': float(delta.mean()),
                'delta_ci95_low': float(np.quantile(bm, 0.025)),
                'delta_ci95_high': float(np.quantile(bm, 0.975)),
                'fraction_events_better_than_copywt': float((delta < 0).mean()),
            })
    return pd.DataFrame(rows)


def manuscript_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = {
        'model': 'Model',
        'ca_rmsd_mean': 'Global Cα RMSD ↓',
        'lddt_mean': 'lDDT ↑',
        'local_ca_rmsd_8a_mean': 'Local Cα RMSD8 ↓',
        'mutation_sidechain_rmsd_mean': 'Mutation-site SC RMSD ↓',
        'mutation_response_rmsd_8a_mean': 'MR-RMSD8 ↓',
    }
    available = [c for c in cols if c in summary.columns]
    return summary[available].rename(columns={c: cols[c] for c in available})


def main():
    p = argparse.ArgumentParser(
        description='Filter already-computed test pair metrics to the hard MutaStruct subset and recompute event-macro metrics.'
    )
    p.add_argument('--project-root', type=Path, default=PROJECT_DEFAULT)
    p.add_argument('--hard-subset-dir', type=Path, default=PROJECT_DEFAULT / 'results/hard_subset_local_ca_gt1')
    p.add_argument('--result', type=Path, action='append', default=[], help='Explicit pair_metrics.csv, directory, or zip. Repeatable.')
    p.add_argument('--no-discover', action='store_true', help='Do not recursively discover pair_metrics.csv under project root.')
    p.add_argument('--output-dir', type=Path, default=PROJECT_DEFAULT / 'results/hard_subset_local_ca_gt1/evaluation')
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    hard_events = pd.read_csv(args.hard_subset_dir / 'hard_subset_events.csv')
    hard_pairs = pd.read_csv(args.hard_subset_dir / 'hard_subset_manifest.csv')
    hard_event_ids = set(hard_events['event_id'].astype(str))
    hard_pair_ids = set(hard_pairs['pair_id'].astype(str))

    sources = list(args.result)
    if not args.no_discover:
        sources.extend(discover_sources(args.project_root))
    # Stable de-duplication of source paths
    seen, source_paths = set(), []
    for s in sources:
        s = Path(s)
        key = str(s.resolve()) if s.exists() else str(s)
        if key not in seen:
            seen.add(key)
            source_paths.append(s)

    source_tables = []
    load_errors = []
    for src in source_paths:
        try:
            source_tables.extend(load_source(src))
        except Exception as exc:
            load_errors.append({'source': str(src), 'error': repr(exc)})

    selected, source_selection, all_sources = choose_model_sources(source_tables, hard_pair_ids, hard_event_ids)

    # Retain only the exact 75 coordinate pairs from the 33 selected events.
    selected = selected[selected['pair_id'].isin(hard_pair_ids) & selected['event_id'].isin(hard_event_ids)].copy()

    # Ensure required metric columns exist so downstream outputs have a stable schema.
    for m in METRICS:
        if m not in selected.columns:
            selected[m] = np.nan

    ev = event_macro(selected)
    summary = make_summary(ev, args.bootstrap, args.seed)
    main = manuscript_table(summary)
    paired = paired_vs_copywt(ev, args.bootstrap, args.seed)

    coverage_rows = []
    for model, g in selected.groupby('model'):
        rec = {
            'model': model,
            'hard_pairs_expected': len(hard_pair_ids),
            'hard_events_expected': len(hard_event_ids),
            'pairs_present': int(g['pair_id'].nunique()),
            'events_present': int(g['event_id'].nunique()),
            'pair_coverage': float(g['pair_id'].nunique() / len(hard_pair_ids)),
            'event_coverage': float(g['event_id'].nunique() / len(hard_event_ids)),
        }
        for m in METRICS:
            rec[f'{m}_event_n'] = int(ev.loc[ev['model'] == model, m].notna().sum()) if m in ev.columns else 0
        coverage_rows.append(rec)
    coverage = pd.DataFrame(coverage_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / 'pair_metrics_hard_subset.csv', index=False)
    ev.to_csv(args.output_dir / 'event_metrics_hard_subset.csv', index=False)
    summary.to_csv(args.output_dir / 'summary_event_macro_with_ci.csv', index=False)
    main.to_csv(args.output_dir / 'main_table_event_macro.csv', index=False)
    paired.to_csv(args.output_dir / 'paired_vs_copywt.csv', index=False)
    coverage.to_csv(args.output_dir / 'prediction_coverage.csv', index=False)
    source_selection.to_csv(args.output_dir / 'source_selection.csv', index=False)
    all_sources.to_csv(args.output_dir / 'discovered_sources.csv', index=False)
    if load_errors:
        pd.DataFrame(load_errors).to_csv(args.output_dir / 'source_load_errors.csv', index=False)

    metadata = {
        'subset_definition': 'event-level median native WT->MT local C-alpha RMSD8 > 1.0 A (or threshold used when subset was built)',
        'hard_events': len(hard_event_ids),
        'hard_pairs': len(hard_pair_ids),
        'aggregation': 'mean coordinate-pair metric within event, then equal-weight mean across mutation events',
        'bootstrap': 'event-level bootstrap of event-macro means',
        'bootstrap_replicates': args.bootstrap,
        'seed': args.seed,
        'note': 'No structure prediction or metric recomputation is performed; existing full-test pair_metrics are filtered and re-aggregated.',
    }
    (args.output_dir / 'evaluation_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    print('\n=== HARD-SUBSET EVENT-MACRO RESULTS ===')
    print(main.to_string(index=False))
    print('\n=== COVERAGE ===')
    print(coverage[['model','pairs_present','events_present','pair_coverage','event_coverage']].to_string(index=False))
    if not paired.empty:
        print('\n=== PAIRED VS COPY-WT (negative delta is better) ===')
        print(paired.to_string(index=False))
    print(f'\nResults written to: {args.output_dir}')


if __name__ == '__main__':
    main()
