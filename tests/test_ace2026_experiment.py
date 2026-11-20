"""CPU regression checks for ACE identity, label isolation and exact replay."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from baselines.ace2026_model import build_ace_svd
from fafirec import ratio_runner as ratio, research_runner as reference
from fafirec.runner import digest, file_digest, json_dump, tensor_digest
from tools import ace2026_experiment as ace


class ACEExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seeds = [17, 18, 19, 20, 21]
        self.records = [{'user': str(i), 'group': ('cold' if i < 2 else 'warm' if i == 5 else 'middle'),
            'train': [1 + i % 2, 3, 4, 2] * 4, 'validation': [5, 6], 'test': [7, 8]} for i in range(6)]
        self.features = torch.randn(13, 8, generator=torch.Generator().manual_seed(40))
        self.features[0].zero_()
        self.identity = {'items': 12, 'user_order': [r['user'] for r in self.records],
            'records_sha256': digest(self.records), 'features_sha256': tensor_digest(self.features)}
        self.base = {**ratio.DEFAULTS, 'max_len': 6, 'device': 'cpu', 'threads': 2}
        master = reference.normalize_config({'reference_main_run': str(self.root / 'reference'),
            'anchor_run': str(self.root / 'anchor'), 'output_dir': str(self.root / 'old_output'),
            'seeds': self.seeds, 'device': 'cpu',
            'variants': [{'name': 'id_ce', 'kind': 'id', 'objective': 'ce'}]})
        self.master_file = self.root / 'master.json'
        json_dump(self.master_file, master)
        self.cfg = ace.normalize_config({'master_config': str(self.master_file),
            'output_dir': str(self.root / 'run'), 'seeds': [17],
            'variants': [{'name': 'ace', 'reg_weight': 50., 'scale': .1, 'lr': .001}],
            'epochs': 2, 'min_epochs': 1, 'eval_interval': 1, 'patience': 4,
            'batch_size': 3, 'eval_batch': 3, 'max_len': 6, 'device': 'cpu', 'threads': 2,
            'd_model': 8, 'n_layers': 1, 'n_heads': 2, 'inner_size': 16,
            'dropout': .1, 'attn_dropout': .1, 'ce_chunk_size': 8})
        self.inputs_patch = patch.object(reference, 'load_anchors', side_effect=self.reference_inputs)
        self.inputs_patch.start()
        self.addCleanup(self.inputs_patch.stop)
        self.cuda_patch = patch('torch.cuda.is_available', return_value=False)
        self.cuda_patch.start()
        self.addCleanup(self.cuda_patch.stop)

    def reference_inputs(self, cfg):
        provenance = {'origin': 'CPU fixture, no checkpoint permitted in ACE initialization',
            'anchors': {str(seed): {'checkpoint_sha256': digest(seed)} for seed in cfg['seeds']}}
        # Deliberately invalid CF checkpoint object. The ACE runner must never
        # inspect it or load it into its model.
        return copy.deepcopy(self.records), self.features, self.identity, self.base, object(), provenance

    def train(self, cfg=None, resume=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return ace.run(cfg or self.cfg, resume)

    def train_grid(self):
        cfg = {**self.cfg, 'seeds': self.seeds, 'output_dir': str(self.root / 'five_seed_run')}
        self.train(cfg)
        paths = [cfg['output_dir']]
        lock = self.root / 'selection_lock.json'
        ace.lock_selection(paths, lock)
        return cfg, paths, lock

    def test_train_isolation_from_cf_and_future_labels(self):
        original = ace.evaluate
        def isolated(model, records, split, *args, **kwargs):
            self.assertEqual(split, 'validation')
            self.assertTrue(all('test' not in r for r in records))
            return original(model, records, split, *args, **kwargs)
        with patch.object(ace, 'evaluate', side_effect=isolated):
            self.train()
        model, _, _, manifest, summary = ace.load_selected(self.cfg['output_dir'], 17, 'ace')
        self.assertIn('never loaded', manifest['initialization'])
        hist = torch.tensor([[0, 0, 1, 3, 4, 2]])
        positives = torch.tensor([[0, 0, 3, 4, 2, 1]])
        model.train()
        loss = model.loss_batch(hist, positives)['loss']
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(summary['optimizer_steps'], 0)
        self.assertEqual(self.train(resume=True)['selected_variant'], 'ace')
        with self.assertRaises(ValueError):
            self.train()

    def test_epoch_boundary_resume_is_bit_exact(self):
        self.train()
        interrupted = {**self.cfg, 'output_dir': str(self.root / 'interrupted')}
        dump = ace.torch_dump
        def interrupt(path, value):
            dump(path, value)
            if Path(path).name == 'last.pt' and value['epoch'] == 1:
                raise RuntimeError('boundary interruption')
        with patch.object(ace, 'torch_dump', side_effect=interrupt), self.assertRaisesRegex(RuntimeError, 'boundary'):
            self.train(interrupted)
        self.train(interrupted, resume=True)
        def best(cfg):
            return torch.load(Path(cfg['output_dir']) / 'seed_17/ace/best.pt', weights_only=True)
        left, right = best(self.cfg), best(interrupted)
        self.assertEqual(left['selected_epoch'], right['selected_epoch'])
        self.assertEqual(left['validation'], right['validation'])
        for key in left['model']:
            torch.testing.assert_close(left['model'][key], right['model'][key], atol=0, rtol=0)

    def test_validation_replay_and_source_artifact_guards(self):
        self.train()
        root = Path(self.cfg['output_dir'])
        result = ace.replay(root, 17, 'ace', 'validation', self.root / 'replay')
        self.assertTrue(result['identical_to_original'])
        self.assertEqual(result['per_user_sha256'], file_digest(root / 'seed_17/ace/validation.csv'))
        with self.assertRaisesRegex(ValueError, 'five-seed'):
            ace.replay(root, 17, 'ace', 'test', self.root / 'unlocked')
        with patch.object(ace, 'source_identity', return_value={'wrong': 'source'}), self.assertRaises(ValueError):
            ace.load_selected(root, 17, 'ace')
        path = root / 'seed_17/ace/best.pt'
        path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact'):
            ace.load_selected(root, 17, 'ace')

    def test_complete_five_seed_lock_and_test_rank_audit(self):
        cfg, paths, lock = self.train_grid()
        validation = ace.aggregate(paths, 'validation', lock, self.root / 'validation.json')
        self.assertEqual(len(validation['aggregates']), 24)
        for seed in self.seeds:
            first = ace.replay(cfg['output_dir'], seed, 'ace', 'test', self.root / f'test/seed_{seed}', lock)
            second = ace.replay(cfg['output_dir'], seed, 'ace', 'test', self.root / f'reload/seed_{seed}', lock)
            self.assertEqual(first['per_user_sha256'], second['per_user_sha256'])
            self.assertEqual(first['metrics'], second['metrics'])
        tested = ace.aggregate(paths, 'test', lock, self.root / 'test.json', self.root / 'test')
        self.assertEqual(tested['selection_lock_sha256'], validation['selection_lock_sha256'])
        self.assertEqual(tested['epoch_selections'], validation['epoch_selections'])
        csv = self.root / 'test/seed_17/per_user.csv'
        csv.write_bytes(csv.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'identity'):
            ace.aggregate(paths, 'test', lock, self.root / 'bad.json', self.root / 'test')

    def test_cache_configuration_and_budget_rejection(self):
        cache = self.root / 'factors.pt'
        torch.save(build_ace_svd(self.features), cache)
        cfg = {**self.cfg, 'svd_cache': str(cache)}
        self.train(cfg)
        bad = torch.load(cache, weights_only=True)
        bad['u'][0, 0] += .1
        torch.save(bad, cache)
        with self.assertRaises(ValueError):
            ace.load_selected(cfg['output_dir'], 17, 'ace')
        for change in ({'unknown': 1}, {'seeds': [17, 17]}, {'variants': [{'name': '../bad'}]}, {'grad_clip': 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ace.normalize_config({**self.cfg, **change})
        other = {**self.cfg, 'output_dir': str(self.root / 'other')}
        self.train(other)
        last_path = Path(other['output_dir']) / 'seed_17/ace/last.pt'
        last = torch.load(last_path, weights_only=True)
        last['logs'][0]['optimizer_steps'] += 1
        ace.torch_dump(last_path, last)
        with self.assertRaisesRegex(ValueError, 'layout'):
            ace.load_selected(other['output_dir'], 17, 'ace')

    def test_lock_rejects_incomplete_grid_and_missing_artifacts(self):
        self.train()
        with self.assertRaisesRegex(ValueError, 'five'):
            ace.lock_selection([self.cfg['output_dir']], self.root / 'incomplete.json')
        cfg, _, lock = self.train_grid()
        data = ace.read(lock)
        data['artifact_sha256'].pop(next(iter(data['artifact_sha256'])))
        data['lock_fingerprint'] = digest({k: v for k, v in data.items() if k != 'lock_fingerprint'})
        json_dump(lock, data)
        with self.assertRaisesRegex(ValueError, 'omits'):
            ace.verify_lock(lock)


if __name__ == '__main__':
    unittest.main()
