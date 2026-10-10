#!/usr/bin/env python3
"""Test checkpoint sanitization on a generated random model, not data."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'geometric_transformer'))
import torch
from network.config import ModelConfig
from network.model import GeometricTransformer
from tools.export_checkpoint import export_checkpoint


class CheckpointExportTests(unittest.TestCase):
    def test_checkpoint_roundtrip_and_metadata_removal(self):
        cfg = ModelConfig(hidden_dim=32, num_heads=4, transformer_layers=1,
                          refinement_layers=1, ffn_multiplier=2, angle_k=3)
        model = GeometricTransformer(cfg)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / 'private.pt'
            torch.save({
                'model_state_dict': model.state_dict(), 'epoch': 7,
                'optimizer_state_dict': {'private_key':'DO_NOT_RELEASE'},
                'metrics': {'pair_id':'DO_NOT_RELEASE'},
                'configuration': {
                    'model_name':'source_model', 'model':asdict(cfg),
                    'graph': {'radius':6., 'max_neighbors':32, 'cache_schema_version':1},
                    'paths': {'manifest':'DO_NOT_RELEASE'},
                    'training': {'learning_rate':.0001, 'seed':42, 'private_ids':['DO_NOT_RELEASE']},
                },
            }, src)
            dst = d / 'public/checkpoints/geometric_transformer_best.pt'
            report = export_checkpoint(src, dst)
            obj = torch.load(dst, weights_only=True)
            self.assertEqual(set(obj), {'model_state_dict','configuration','epoch'})
            self.assertNotIn('paths', obj['configuration'])
            self.assertNotIn('private_ids', obj['configuration']['training'])
            self.assertNotIn('DO_NOT_RELEASE', json.dumps(obj['configuration']))
            other = GeometricTransformer(ModelConfig(**obj['configuration']['model']))
            other.load_state_dict(obj['model_state_dict'], strict=True)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, other.state_dict()[k]))
            self.assertEqual(report['epoch'], 7)
            with self.assertRaises(FileExistsError):
                export_checkpoint(src, dst)

    def test_nonmatching_architecture_rejected(self):
        cfg = ModelConfig()
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);src=d/'other.pt'
            torch.save({'model_state_dict':{'wrong_layer':torch.zeros(4)},
                        'configuration':{'model':asdict(cfg)}},src)
            with self.assertRaises(RuntimeError):
                export_checkpoint(src,d/'release/model.pt')
            self.assertFalse((d/'release/model.pt').exists())

    def test_no_other_architecture_folders(self):
        for p in ROOT.rglob('*'):
            if p.is_dir():
                self.assertNotIn(p.name, {'differential', 'frame', 'local'})

if __name__ == '__main__':
    unittest.main(verbosity=2)
