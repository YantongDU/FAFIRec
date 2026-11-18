"""CPU fixtures for isolated labels, immutable anchors, resume and replay."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import analyze_research
import evaluate_research
from fafirec import ratio_runner as ratio, research_runner as research, validation_runner as old
from fafirec.data import file_sha256
from fafirec.runner import file_digest, json_dump


class ResearchRunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        data = self.root / 'data'
        data.mkdir()
        items = [{'id': i, 'raw_id': str(i), 'text': f'static movie {i}'} for i in range(1, 13)]
        records = [{'user': str(i), 'train': [1 + i % 2, 3, 4, 2] * 4,
                    'validation': [5, 6], 'test': [7, 8]} for i in range(6)]
        for name, values in (('items.jsonl', items), ('interactions.jsonl', records)):
            (data / name).write_text('\n'.join(map(json.dumps, values)) + '\n', encoding='utf-8')
        json_dump(data / 'manifest.json', {'format_version': 2, 'split': {'ratios': [.8, .1, .1]},
            'files': {name: file_sha256(data / name) for name in ('items.jsonl', 'interactions.jsonl')}})
        main, anchor = self.root / 'main', self.root / 'anchor'
        base = {**ratio.DEFAULTS, 'data_dir': str(data), 'output_dir': str(main),
            'feature_mode': 'lexical', 'embedding_path': None, 'lexical_dim': 8,
            'seeds': [17], 'controls': ['cf', 'uniform', 'diagnostic'], 'epochs': 1, 'teacher_epochs': 1,
            'patience': 4, 'batch_size': 3, 'eval_batch': 3, 'd_model': 8, 'max_len': 6,
            'n_heads': 2, 'n_layers': 1, 'dropout': .1, 'device': 'cpu', 'threads': 2}
        search = {'reference_run_dir': str(main), 'output_dir': str(anchor), 'seeds': [17],
            'epochs': 1, 'patience': 4, 'device': 'cpu', 'threads': 2,
            'variants': [{'name': 'cf_lr1e3', 'control': 'cf', 'lr': .001}]}
        with contextlib.redirect_stdout(io.StringIO()):
            ratio.run(base)
            old.run(search)
        mm, sm = research._read(main / 'manifest.json'), research._read(anchor / 'manifest.json')
        lock = self.root / 'lock.json'
        json_dump(lock, {'selected': 'cf_lr1e3', 'selection_split': 'validation',
            'new_test_metrics_used_for_baseline_selection': False,
            'main_fingerprint': mm['fingerprint'], 'search_fingerprint': sm['fingerprint'],
            'artifact_sha256': {str(main / 'manifest.json'): file_digest(main / 'manifest.json'),
                                str(anchor / 'manifest.json'): file_digest(anchor / 'manifest.json')}})
        checkpoint_path = anchor / 'seed_17/cf_lr1e3/best.pt'
        (main / 'analysis').mkdir()
        json_dump(main / 'analysis/strong_cf_comparison.json', {'complete': True,
            'selection': {'selected': 'cf_lr1e3', 'lock_sha256': file_digest(lock)},
            'source': {'selection_lock_file': str(lock)},
            'per_seed': [{'seed': 17, 'checkpoint_sha256': file_digest(checkpoint_path)}]})
        self.cfg = research.normalize_config({'reference_main_run': str(main), 'anchor_run': str(anchor),
            'output_dir': str(self.root / 'research'), 'seeds': [17], 'epochs': 2, 'min_epochs': 1,
            'patience': 4, 'eval_interval': 1, 'batch_size': 3, 'eval_batch': 3,
            'alpha_grid': [0, 1], 'ce_chunk_size': 8, 'device': 'cpu', 'threads': 2,
            'probe_negatives': 2, 'probe_window': 2, 'router_epochs': 2, 'router_min_epochs': 1,
            'variants': [{'name': 'text_bpr', 'kind': 'text', 'objective': 'bpr'}]})

    def tearDown(self):
        self.temporary.cleanup()

    def run_research(self, cfg=None, resume=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return research.run(cfg or self.cfg, resume)

    def test_anchor_identity_and_zero_residual_exact(self):
        records, features, _, _, anchors, provenance = research.load_anchors(self.cfg)
        model = research._new_model(anchors[17], features, [{k: v for k, v in r.items() if k != 'test'}
            for r in records], self.cfg['variants'][0], self.cfg, 17, torch.device('cpu'))
        model.eval().set_calibration(0., 'global')
        history = torch.tensor([[0, 0, 1, 3, 4, 2]])
        torch.testing.assert_close(model.user_vectors(history), model.anchor.user_vectors(history), rtol=0, atol=0)
        torch.testing.assert_close(model.item_vectors(), model.anchor.item_vectors(), rtol=0, atol=0)
        self.assertFalse(provenance['old_diagnostics_used_as_features'])
        path = Path(self.cfg['anchor_run']) / 'seed_17/cf_lr1e3/best.pt'
        path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum differs'):
            research.load_anchors(self.cfg)

    def test_test_isolation_reload_and_independent_analysis(self):
        original, calls = research.evaluate, []
        def isolated(model, records, split, *args, **kwargs):
            self.assertEqual(split, 'validation')
            self.assertTrue(all('test' not in r for r in records))
            calls.append(split)
            return original(model, records, split, *args, **kwargs)
        with patch.object(research, 'evaluate', side_effect=isolated):
            selection = self.run_research()
        self.assertTrue(calls)
        self.assertEqual(selection['split'], 'validation')
        root = Path(self.cfg['output_dir'])
        model, records, diagnostics, cfg, manifest = research.load_selected(root, 17, 'text_bpr')
        self.assertTrue(all('test' in r for r in records))
        rows, metrics = original(model, records, 'validation', diagnostics, cfg, 17)
        saved = research._read(root / 'seed_17/text_bpr/expert/summary.json')
        self.assertEqual(metrics, saved['validation'])
        self.assertFalse(manifest['test_evaluated_by_training_entrypoint'])
        replay = evaluate_research.replay(root, 17, 'text_bpr', 'expert', 'validation', self.root / 'replayed')
        self.assertTrue(replay['identical_to_original'])
        audit = analyze_research.analyze(root)
        self.assertEqual(audit['job_count'], 1)
        self.assertTrue(audit['all_artifact_hashes_verified'])
        self.assertEqual(self.run_research(resume=True), selection)
        with self.assertRaises(FileExistsError):
            self.run_research()
        csv = root / 'seed_17/text_bpr/expert/validation.csv'
        csv.write_bytes(csv.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'artifacts changed'):
            research.load_selected(root, 17, 'text_bpr')

    def test_epoch_boundary_resume_is_exact(self):
        self.run_research()
        interrupted = {**self.cfg, 'output_dir': str(self.root / 'interrupted')}
        original = research.torch_dump
        def stop_after_boundary(path, value):
            original(path, value)
            if Path(path).name == 'last.pt' and value['epoch'] == 1:
                raise RuntimeError('simulated epoch boundary interruption')
        with patch.object(research, 'torch_dump', side_effect=stop_after_boundary), self.assertRaisesRegex(RuntimeError, 'simulated'):
            self.run_research(interrupted)
        self.run_research(interrupted, True)
        def selected(root):
            return torch.load(Path(root) / 'seed_17/text_bpr/expert/best.pt', map_location='cpu', weights_only=True)
        full, resumed = selected(self.cfg['output_dir']), selected(interrupted['output_dir'])
        self.assertEqual(full['calibration'], resumed['calibration'])
        self.assertEqual(full['selected_epoch'], resumed['selected_epoch'])
        for key in full['model']:
            torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)

    def test_router_stage_is_separate_and_replayable(self):
        cfg = {**self.cfg, 'train_router': True}
        self.run_research(cfg)
        root = Path(cfg['output_dir'])
        self.assertTrue((root / 'seed_17/text_bpr/expert/best.pt').is_file())
        self.assertTrue((root / 'seed_17/text_bpr/router/best.pt').is_file())
        model, records, diagnostics, local, _ = research.load_selected(root, 17, 'text_bpr', 'router')
        _, value = ratio.evaluate(model, records, 'validation', diagnostics, local, 17)
        self.assertEqual(value, research._read(root / 'seed_17/text_bpr/router/summary.json')['validation'])
        self.assertEqual(analyze_research.analyze(root)['job_count'], 2)

    def test_ce_and_configuration_guards(self):
        cfg = {**self.cfg, 'variants': [{'name': 'id_ce', 'kind': 'id', 'objective': 'ce', 'score_scale': 8.}]}
        self.run_research(cfg)
        self.assertEqual(analyze_research.analyze(cfg['output_dir'])['job_count'], 1)
        for update in ({'epochs': True}, {'min_epochs': 3}, {'alpha_grid': [1.]},
                       {'seeds': [17, 17]}, {'train_router': 1}, {'routes': ['relative']},
                       {'output_dir': self.cfg['anchor_run']}, {'unknown': 1}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                research.normalize_config({**self.cfg, **update})
        for update in ({'name': '../escape'}, {'objective': 'invalid'}, {'lr': False}, {'score_scale': 0}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                research.normalize_config({**self.cfg, 'variants': [{**self.cfg['variants'][0], **update}]})


if __name__ == '__main__':
    unittest.main()
