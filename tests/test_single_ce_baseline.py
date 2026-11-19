"""CPU coverage for independent single-branch CE continuation and test locks."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from fafirec import ratio_runner as ratio, research_runner as research
from fafirec.ml1m_model import PaperFAFIRecModel
from fafirec.runner import cpu_state, digest, file_digest, json_dump, tensor_digest
from tools import single_ce_baseline as single


class SingleCEBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.seeds = [17, 18, 19, 20, 21]
        self.records = [{'user': str(i), 'group': ('cold' if i < 2 else 'warm' if i == 5 else 'middle'),
                         'train': [1 + i % 2, 3, 4, 2] * 4, 'validation': [5, 6], 'test': [7, 8]}
                        for i in range(6)]
        self.features = torch.randn(13, 8, generator=torch.Generator().manual_seed(40))
        self.features[0].zero_()
        self.base = {**ratio.DEFAULTS, 'd_model': 8, 'max_len': 6, 'n_heads': 2,
                     'n_layers': 1, 'dropout': .1, 'device': 'cpu', 'threads': 2}
        self.identity = {'items': 12, 'user_order': [r['user'] for r in self.records],
                         'records_sha256': digest(self.records), 'features_sha256': tensor_digest(self.features)}
        self.anchors = {}
        for seed in self.seeds:
            torch.manual_seed(seed)
            cfg = ratio.model_config(self.base, 'cf', 12)
            model = PaperFAFIRecModel(text_features=self.features, **cfg)
            self.anchors[seed] = {'model': cpu_state(model), 'model_config': cfg}
        self.master = research.normalize_config({'reference_main_run': str(self.root / 'main'),
            'anchor_run': str(self.root / 'anchor'), 'output_dir': str(self.root / 'core'),
            'seeds': self.seeds, 'epochs': 2, 'min_epochs': 1, 'patience': 4,
            'eval_interval': 1, 'batch_size': 3, 'eval_batch': 3, 'ce_chunk_size': 8,
            'device': 'cpu', 'threads': 2,
            'variants': [{'name': 'id_ce', 'kind': 'id', 'objective': 'ce', 'score_scale': 16.}]})
        self.master_file = self.root / 'master.json'
        json_dump(self.master_file, self.master)
        self.replica_root = self.root / 'single'
        self.input_patch = patch.object(single, '_inputs', side_effect=self.inputs)
        self.input_patch.start()
        self.addCleanup(self.input_patch.stop)
        self.cuda_patch = patch('torch.cuda.is_available', return_value=False)
        self.cuda_patch.start()
        self.addCleanup(self.cuda_patch.stop)

    def inputs(self, master, seed):
        anchor = self.anchors[seed]
        provenance = {'old_diagnostics_used_as_features': False, 'shared_anchor_origin': 'CPU fixture',
                      'anchors': {str(seed): {'checkpoint_sha256': digest({'seed': seed}),
                          'tensor_sha256': digest({k: tensor_digest(v) for k, v in anchor['model'].items()})}}}
        return copy.deepcopy(self.records), self.features, self.identity, self.base, {seed: anchor}, provenance

    def train(self, seed=17, root=None, resume=False):
        out = (root or self.replica_root) / f'seed_{seed}'
        with contextlib.redirect_stdout(io.StringIO()):
            return single.run(self.master_file, seed, out, resume)

    def train_grid(self):
        for seed in self.seeds:
            self.train(seed)
        lock_path = self.root / 'single_lock.json'
        single.lock_selection(self.master_file, self.replica_root, lock_path)
        return lock_path

    def test_dense_loss_gradients_and_immutable_anchor(self):
        anchor = PaperFAFIRecModel(text_features=self.features, **self.anchors[17]['model_config'])
        anchor.load_state_dict(self.anchors[17]['model'])
        original = cpu_state(anchor)
        model = single.SingleCEBaseline(anchor, ce_chunk_size=3).eval()
        self.assertFalse(any(p.requires_grad for name, p in model.cf.named_parameters()
                             if not name.startswith(single.TRAINABLE_PREFIXES)))
        self.assertTrue(all(p.requires_grad for name, p in model.cf.named_parameters()
                            if name.startswith(single.TRAINABLE_PREFIXES)))
        hist = torch.tensor([[0, 1, 3], [0, 2, 4]])
        pos = torch.tensor([[0, 3, 2], [0, 4, 1]])
        seen = torch.zeros(2, 13, dtype=torch.bool)
        seen[:, [1, 2, 3, 4]] = True
        valid = hist.ne(0) & pos.ne(0)
        states = model.cf._encode_tokens(model.cf.item_id_table(hist), hist.ne(0))
        logits = 16. * (states[valid] @ model.item_vectors().T)
        owners = torch.arange(2).unsqueeze(1).expand_as(hist)[valid]
        blocked = seen[owners].clone()
        blocked[torch.arange(len(owners)), pos[valid]] = False
        blocked[:, 0] = True
        dense = F.cross_entropy(logits.masked_fill(blocked, -torch.inf), pos[valid])
        parameters = [p for p in model.parameters() if p.requires_grad]
        dense_grads = torch.autograd.grad(dense, parameters)
        chunked = model.loss_batch(hist, pos, seen)
        chunked_grads = torch.autograd.grad(chunked, parameters)
        torch.testing.assert_close(chunked, dense, rtol=1e-6, atol=1e-6)
        for left, right in zip(dense_grads, chunked_grads):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        optimizer = torch.optim.Adam(parameters, lr=.001, weight_decay=1e-5)
        optimizer.zero_grad()
        model.loss_batch(hist, pos, seen).backward()
        torch.testing.assert_close(model.cf.item_id_table.weight.grad[0], torch.zeros(8), rtol=0, atol=0)
        optimizer.step()
        self.assertFalse(torch.equal(model.cf.item_id_table.weight, original['item_id_table.weight']))
        for name, value in anchor.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
            if not name.startswith(single.TRAINABLE_PREFIXES):
                torch.testing.assert_close(model.cf.state_dict()[name], value, rtol=0, atol=0)
        for kwargs in ({'score_scale': 0}, {'score_scale': float('nan')}, {'score_scale': True},
                       {'ce_chunk_size': 0}, {'ce_chunk_size': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                single.SingleCEBaseline(anchor, **kwargs)

    def test_train_isolation_reload_and_identity_guards(self):
        evaluate = single.evaluate
        def isolated(model, records, split, *args, **kwargs):
            self.assertEqual(split, 'validation')
            self.assertTrue(all('test' not in r for r in records))
            return evaluate(model, records, split, *args, **kwargs)
        with patch.object(single, 'evaluate', side_effect=isolated):
            summary = self.train()
        out = self.replica_root / 'seed_17'
        result = single.replay(out, 17, 'validation', self.root / 'replay')
        self.assertTrue(result['identical_to_original'])
        self.assertEqual(result['metrics'], summary['validation'])
        self.assertEqual(self.train(resume=True), summary)
        with self.assertRaises(FileExistsError):
            self.train()
        with self.assertRaisesRegex(ValueError, 'selection-lock'):
            single.replay(out, 17, 'test', self.root / 'unlocked_test')
        with self.assertRaisesRegex(ValueError, 'identity'):
            single.load_selected(out, 18)
        with patch.object(single, 'source_identity', return_value={'changed': 'source'}), self.assertRaises(ValueError):
            single.load_selected(out, 17)
        checkpoint = out / 'best.pt'
        checkpoint.write_bytes(checkpoint.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'artifacts changed'):
            single.load_selected(out, 17)

    def test_epoch_boundary_resume_is_bit_exact(self):
        self.train()
        interrupted_root = self.root / 'interrupted'
        original = single.torch_dump
        def interrupt(path, value):
            original(path, value)
            if Path(path).name == 'last.pt' and value['epoch'] == 1:
                raise RuntimeError('simulated boundary interruption')
        with patch.object(single, 'torch_dump', side_effect=interrupt), self.assertRaisesRegex(RuntimeError, 'simulated'):
            self.train(root=interrupted_root)
        self.train(root=interrupted_root, resume=True)
        def state(folder):
            return torch.load(folder / 'seed_17/best.pt', map_location='cpu', weights_only=True)
        full, resumed = state(self.replica_root), state(interrupted_root)
        self.assertEqual(full['selected_epoch'], resumed['selected_epoch'])
        self.assertEqual(full['validation'], resumed['validation'])
        for name, value in full['model'].items():
            torch.testing.assert_close(value, resumed['model'][name], rtol=0, atol=0)
        # The authoritative last checkpoint can be ahead of a readable log
        # after interruption; a resumed selected run must still be verifiable.
        self.assertEqual(single.load_selected(interrupted_root / 'seed_17', 17)[4]['epochs_completed'], 2)

    def test_five_seed_lock_test_replays_and_independent_rank_audit(self):
        lock = self.train_grid()
        validated = single.aggregate(self.replica_root, 'validation', self.root / 'combined_validation.json', lock)
        self.assertEqual(len(validated['aggregates']), 24)
        self.assertEqual(validated['seed_ids'], self.seeds)
        self.assertTrue(validated['all_artifact_hashes_verified'])
        for seed in self.seeds:
            out = self.replica_root / f'seed_{seed}'
            first = single.replay(out, seed, 'test', self.root / f'test/seed_{seed}', lock)
            second = single.replay(out, seed, 'test', self.root / f'reloaded/seed_{seed}', lock)
            self.assertEqual(first['metrics'], second['metrics'])
            self.assertEqual(first['per_user_sha256'], second['per_user_sha256'])
        tested = single.aggregate(self.replica_root, 'test', self.root / 'combined_test.json', lock, self.root / 'test')
        self.assertEqual(tested['selection_lock_sha256'], validated['selection_lock_sha256'])
        self.assertEqual(tested['epoch_selections'], validated['epoch_selections'])
        with self.assertRaises(FileExistsError):
            single.lock_selection(self.master_file, self.replica_root, lock)
        csv = self.root / 'test/seed_17/per_user.csv'
        csv.write_bytes(csv.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'assets changed'):
            single.aggregate(self.replica_root, 'test', self.root / 'bad_test.json', lock, self.root / 'test')

    def test_lock_completeness_configuration_and_training_budget_guards(self):
        self.train()
        with self.assertRaisesRegex(ValueError, 'seed directories'):
            single.lock_selection(self.master_file, self.replica_root, self.root / 'incomplete.json')
        for seed in self.seeds[1:]:
            self.train(seed)
        extra = self.replica_root / 'seed_22'
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, 'extra'):
            single.lock_selection(self.master_file, self.replica_root, self.root / 'extra.json')
        extra.rmdir()
        lock = self.root / 'lock.json'
        single.lock_selection(self.master_file, self.replica_root, lock)
        changed_master = {**self.master, 'lr_unused': .1}
        # Unknown config keys must be rejected before the lock is consumed.
        json_dump(self.master_file, changed_master)
        with self.assertRaises(ValueError):
            single.verify_lock(lock, self.replica_root / 'seed_17', 17)
        json_dump(self.master_file, self.master)
        last_path = self.replica_root / 'seed_17/last.pt'
        last = torch.load(last_path, map_location='cpu', weights_only=True)
        last['logs'][0]['optimizer_steps'] += 1
        single.torch_dump(last_path, last)
        with self.assertRaisesRegex(ValueError, 'history differs'):
            single.load_selected(self.replica_root / 'seed_17', 17)
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            single.verify_lock(lock, self.replica_root / 'seed_17', 17)


if __name__ == '__main__':
    unittest.main()
