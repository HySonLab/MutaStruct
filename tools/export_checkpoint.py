#!/usr/bin/env python3
"""Export a *selected Qin-style* checkpoint without optimizer, paths, or labels.

This does not choose a checkpoint, grant redistribution rights, or prove that the
weights are the ones used in a manuscript. Only load a trusted local checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / 'geometric_transformer'
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def primitive(value: Any) -> Any:
    if value is None or type(value) in (bool, int, str):
        if isinstance(value, str) and (len(value) > 128 or '/' in value or '\\' in value):
            raise ValueError('Unexpected path/long string in whitelisted configuration value')
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f'Unsupported configuration type: {type(value).__name__}')


def numeric(value: Any) -> bool:
    return type(value) in (int, float, bool) and math.isfinite(float(value))


def sanitize_config(config: dict) -> dict:
    """Allowlist architecture settings; never copy a paths/data/IDs dictionary."""
    from network.config import ModelConfig
    from common.graph import GraphBuildConfig
    from common.losses import LossWeights
    from common.trainer import TrainingConfig

    if not isinstance(config, dict) or not isinstance(config.get('model'), dict):
        raise ValueError('Checkpoint needs configuration.model; pass --run-config for a separate configuration.json')
    saved = config['model']
    # Characteristic fields distinguish the supplied architecture from older GTs.
    if not {'transformer_layers', 'sigma_angle_degrees', 'angle_k'} <= set(saved):
        raise ValueError('Not an explicitly configured Qin-style checkpoint. Other GT variants are not exportable here.')
    valid = {f.name for f in fields(ModelConfig)}
    unknown = set(saved) - valid
    if unknown:
        raise ValueError(f'Unknown model settings {sorted(unknown)}: use the matching model source rather than discard them')
    model = ModelConfig(**{k: primitive(v) for k, v in saved.items()})
    if model.angle_reduction not in ('max', 'mean'):
        raise ValueError('Unsupported angle reduction')
    graph_saved = config.get('graph', {})
    valid_graph = {f.name for f in fields(GraphBuildConfig)}
    graph = GraphBuildConfig(**{k: primitive(v) for k, v in graph_saved.items() if k in valid_graph})
    out = {'model_name': 'geometric_transformer', 'model': asdict(model), 'graph': asdict(graph)}

    # Keep only numeric documented settings, plus a small explicit enum list.
    for section, typ in [('training', TrainingConfig), ('loss', LossWeights)]:
        allowed = {f.name for f in fields(typ)}
        out[section] = {}
        for k, v in config.get(section, {}).items():
            if k not in allowed:
                continue
            if numeric(v):
                out[section][k] = primitive(v)
            elif section == 'training' and k == 'monitor' and v in (
                'loss_total','ca_rmsd','local_rmsd','local_ca_rmsd','all_atom_rmsd',
                'mr_rmsd_8a','mr_excess_8a','mutation_sidechain_rmsd',
            ):
                out[section][k] = v
    if 'local_radius' in config and numeric(config['local_radius']):
        out['local_radius'] = primitive(config['local_radius'])
    if config.get('pair_class') in (None, 'all','canonical_wt','background_matched','mapped_core_length_mismatch'):
        out['pair_class'] = config.get('pair_class', 'all')
    return out


def export_checkpoint(source: Path, destination: Path, run_config: Path | None = None,
                      trust_checkpoint: bool = False) -> dict:
    import torch
    from network.config import ModelConfig
    from network.model import GeometricTransformer

    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}; choose a new output path')
    try:
        obj = torch.load(source, map_location='cpu', weights_only=True)
    except Exception as exc:
        if not trust_checkpoint:
            raise RuntimeError(
                'Restricted checkpoint loading failed. For your own trusted local training checkpoint only, '
                'retry with --trust-checkpoint. This permits pickle loading and can execute code.'
            ) from exc
        obj = torch.load(source, map_location='cpu', weights_only=False)
    if not isinstance(obj, dict) or 'model_state_dict' not in obj:
        raise ValueError('Expected a training checkpoint with model_state_dict')
    config = obj.get('configuration', {})
    if run_config is not None:
        supplied = json.loads(run_config.read_text(encoding='utf-8'))
        if config.get('model') and config['model'] != supplied.get('model'):
            raise ValueError('External run configuration disagrees with checkpoint architecture')
        config = supplied
    clean_config = sanitize_config(config)
    state = obj['model_state_dict']
    if not isinstance(state, dict) or not state:
        raise ValueError('Empty/non-dictionary model state')
    clean_state = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError('State dictionary must contain only string-to-tensor entries')
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f'Non-finite model weight: {key}')
        clean_state[key] = value.detach().cpu().contiguous().clone()
    model = GeometricTransformer(ModelConfig(**clean_config['model']))
    model.load_state_dict(clean_state, strict=True)  # Reject other architectures, including legacy GTs.
    epoch = obj.get('epoch', -1)
    if type(epoch) not in (int, float) or not math.isfinite(float(epoch)):
        epoch = -1
    clean = {'model_state_dict': clean_state, 'configuration': clean_config, 'epoch': int(epoch)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    try:
        torch.save(clean, temporary)
        round_trip = torch.load(temporary, map_location='cpu', weights_only=True)
        for k in clean_state:
            if not torch.equal(clean_state[k], round_trip['model_state_dict'][k]):
                raise RuntimeError(f'Round-trip weight mismatch for {k}')
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata = {
        'architecture': 'Geometric Transformer (Qin-style geometric attention)',
        'epoch': int(epoch),
        'source_checkpoint_sha256': sha256_file(source),
        'released_checkpoint_sha256': sha256_file(destination),
        'released_bytes': destination.stat().st_size,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'export_format': 'model_state_dict + whitelisted configuration + epoch',
        'omitted': ['optimizer','scheduler','gradient_scaler','RNG','metrics','paths','dataset records','unrecognized metadata'],
        'warning': 'Architecture compatibility is verified; manuscript provenance and redistribution authorization require author review.',
    }
    (destination.parent / 'release_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    configs = destination.parent.parent / 'configs'
    configs.mkdir(exist_ok=True)
    (configs / 'released_run.json').write_text(json.dumps(clean_config, indent=2)+'\n')
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', type=Path, default=ROOT/'checkpoints/geometric_transformer_best.pt')
    p.add_argument('--run-config', type=Path)
    p.add_argument('--trust-checkpoint', action='store_true', help='Permit pickle loading of your trusted local checkpoint only')
    a = p.parse_args()
    print(json.dumps(export_checkpoint(a.checkpoint, a.output, a.run_config, a.trust_checkpoint), indent=2))

if __name__ == '__main__':
    main()
