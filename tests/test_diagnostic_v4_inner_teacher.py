"""Train-internal label isolation and exact fixed-budget teacher resume."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from baselines.ace2026_model import build_ace_svd
from experiments.diagnostic_v4 import inner_teacher as teacher
from fafirec.runner import digest


class InnerTeacherTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = [{'user': str(i), 'group': 'middle',
            'train': [1 + (j + i) % 9 for j in range(8)],
            'validation': [11], 'test': [12]} for i in range(6)]
        self.features = torch.randn(13, 8, generator=torch.Generator().manual_seed(77))
        self.features[0].zero_()
        self.factors = build_ace_svd(self.features)
        self.small_model = {**teacher.MODEL, 'd_model': 8, 'max_len': 6,
            'n_layers': 1, 'inner_size': 16, 'dropout': .1, 'attn_dropout': .1,
            'ce_chunk_size': 8}
        self.cfg = dict(seed=17, epochs=2, output_dir=str(self.root / 'full'),
            device='cpu', threads=2, reference_run=str(self.root / 'reference'))
        self.patches = [patch.object(teacher, 'MODEL', self.small_model),
            patch.object(teacher, 'prepare_inputs', side_effect=self.inputs),
            patch('torch.cuda.is_available', return_value=False),
            patch.object(teacher.ace, 'evaluate', side_effect=AssertionError('Teacher must never evaluate holdouts'))]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def inputs(self, _cfg):
        inner, held, split = teacher.split_inner_records(copy.deepcopy(self.records))
        return inner, held, self.features, self.factors, split, {'fixture': 'no fitted reference weights'}

    def run_teacher(self, cfg=None, resume=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return teacher.run(cfg or self.cfg, resume)

    def test_split_is_only_train_and_partitions_positions(self):
        records = [{'user': str(n), 'train': list(range(1, n + 1)),
            'validation': [999], 'test': [998]} for n in (2, 3, 8, 60)]
        original = copy.deepcopy(records)
        inner, held, info = teacher.split_inner_records(records)
        self.assertEqual(records, original)
        self.assertEqual([len(r['calibration']) for r in held], [0, 1, 1, 10])
        for a, b, old in zip(inner, held, records):
            self.assertEqual(set(a), {'user', 'train'})
            self.assertEqual(a['train'], b['prefix'])
            self.assertEqual(a['train'] + b['calibration'], old['train'])
            self.assertGreaterEqual(len(a['train']), 2)
        changed = copy.deepcopy(records)
        for row in changed:
            row['validation'], row['test'] = [12345], [12346]
            row['untrusted_extra'] = 'not a training input'
        self.assertEqual((inner, held, info), teacher.split_inner_records(changed))
        changed[2]['train'][-1] = 22
        changed_inner, _, changed_info = teacher.split_inner_records(changed)
        self.assertEqual(inner, changed_inner)
        self.assertNotEqual(info['calibration_records_sha256'], changed_info['calibration_records_sha256'])

    def test_fixed_budget_has_no_holdout_selection_and_load_is_frozen(self):
        self.run_teacher()
        model, inner, held, manifest = teacher.load_teacher(self.cfg['output_dir'])
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertTrue(all(set(r) <= {'user', 'group', 'train'} for r in inner))
        self.assertTrue(all(r['calibration'] for r in held))
        self.assertFalse(manifest['calibration_used_for_training_or_early_stop'])
        folder = Path(self.cfg['output_dir'])
        saved = torch.load(folder / 'teacher.pt', weights_only=True)
        self.assertEqual(saved['epoch'], 2)
        self.assertEqual(len(teacher.ace.read(folder / 'training.json')), 2)
        before = (folder / 'teacher.pt').read_bytes()
        self.run_teacher(resume=True)
        self.assertEqual(before, (folder / 'teacher.pt').read_bytes())
        with self.assertRaisesRegex(ValueError, 'Output exists'):
            self.run_teacher()

    def test_epoch_boundary_resume_matches_uninterrupted_model_exactly(self):
        self.run_teacher()
        interrupted = {**self.cfg, 'output_dir': str(self.root / 'interrupted')}
        original = teacher.torch_dump
        def stop_after_checkpoint(path, value):
            original(path, value)
            if Path(path).name == 'last.pt' and value['epoch'] == 1:
                raise RuntimeError('deliberate interruption')
        with patch.object(teacher, 'torch_dump', side_effect=stop_after_checkpoint), \
                self.assertRaisesRegex(RuntimeError, 'deliberate'):
            self.run_teacher(interrupted)
        self.run_teacher(interrupted, resume=True)
        full = torch.load(Path(self.cfg['output_dir']) / 'teacher.pt', weights_only=True)
        recovered = torch.load(Path(interrupted['output_dir']) / 'teacher.pt', weights_only=True)
        for key, value in full['model'].items():
            torch.testing.assert_close(value, recovered['model'][key], rtol=0, atol=0)
        self.assertEqual(teacher.ace.read(Path(self.cfg['output_dir']) / 'training.json'),
                         teacher.ace.read(Path(interrupted['output_dir']) / 'training.json'))
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            self.run_teacher({**interrupted, 'epochs': 3}, resume=True)

    def test_rejects_future_fields_source_changes_and_artifact_tampering(self):
        self.run_teacher()
        with patch.object(teacher, 'source_identity', return_value={'changed': 'source'}), \
                self.assertRaisesRegex(ValueError, 'source changed'):
            teacher.load_teacher(self.cfg['output_dir'])
        path = Path(self.cfg['output_dir']) / 'training.json'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            teacher.load_teacher(self.cfg['output_dir'])
        inner, _, split = teacher.split_inner_records(self.records)
        inner[0]['validation'] = [11]
        split['inner_train_records_sha256'] = digest(inner)
        manifest = {'config': self.cfg, 'fingerprint': 'fixture', 'split': split, 'batch_size': 3}
        model = teacher.ACE2026Model(self.features, svd_factors=self.factors, **self.small_model)
        with self.assertRaisesRegex(ValueError, 'prefix-only'):
            teacher.fit_teacher(model, inner, manifest, self.root / 'unsafe')


if __name__ == '__main__':
    unittest.main()
