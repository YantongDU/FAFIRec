import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from fafirec.data import (TrainingDataset, assign_training_groups, evaluation_batch,
                          load_bundle, pad_history)
from prepare_data import build_parser, prepare


def dump_jsonl(path, values):
    path.write_text(''.join(json.dumps(value) + '\n' for value in values), encoding='utf-8')


def update_hashes(root):
    manifest_path = root / 'manifest.json'
    value = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'format_version': 1, 'dataset': 'fixture'}
    value['files'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in ('items.jsonl', 'interactions.jsonl')}
    manifest_path.write_text(json.dumps(value), encoding='utf-8')


class PreparedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.items = [{'id': i, 'raw_id': f'i{i}', 'text': f'metadata word{i}'} for i in range(1, 7)]
        self.records = [{'user': 'a', 'train': [1, 2], 'validation': 3, 'test': 4},
                        {'user': 'b', 'train': [2, 3, 4], 'validation': 5, 'test': 6}]
        dump_jsonl(self.root / 'items.jsonl', self.items)
        dump_jsonl(self.root / 'interactions.jsonl', self.records)
        update_hashes(self.root)

    def embeddings(self):
        path = self.root / 'features.pt'
        torch.manual_seed(7)
        features = torch.randn(7, 4)
        features[0] = 0
        torch.save(features, path)
        manifest = {'row_ids': ['<PAD>'] + [r['raw_id'] for r in self.items],
                    'embedding_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        path.with_suffix('.manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return path, features, manifest

    def test_features_require_explicit_mapping_without_silent_lexical_fallback(self):
        with self.assertRaisesRegex(ValueError, 'requires embedding_path'):
            load_bundle(self.root)
        path, expected, manifest = self.embeddings()
        records, features, info = load_bundle(self.root, embedding_path=path)
        torch.testing.assert_close(features, expected)
        self.assertEqual([r['group'] for r in records], ['cold', 'warm'])
        self.assertTrue(info['embedding_mapping_verified'])
        manifest['row_ids'][1:3] = list(reversed(manifest['row_ids'][1:3]))
        path.with_suffix('.manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'row-ID'):
            load_bundle(self.root, embedding_path=path)

    def test_changed_artifact_missing_manifest_and_padding_are_rejected(self):
        path, features, manifest = self.embeddings()
        torch.save(features + 1, path)
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            load_bundle(self.root, embedding_path=path)
        manifest['embedding_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix('.manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'padding row zero'):
            load_bundle(self.root, embedding_path=path)
        path.with_suffix('.manifest.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing feature mapping'):
            load_bundle(self.root, embedding_path=path)

    def test_data_tampering_is_rejected_and_lexical_is_deterministic(self):
        a, x, ma = load_bundle(self.root, feature_mode='lexical', lexical_dim=16)
        b, y, mb = load_bundle(self.root, feature_mode='lexical', lexical_dim=16)
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertEqual(x[0].count_nonzero().item(), 0)
        self.assertIn('NOT LLM', ma['feature_type'])
        self.assertEqual(ma['feature_tensor_sha256'], mb['feature_tensor_sha256'])
        with (self.root / 'interactions.jsonl').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'Data file SHA-256'):
            load_bundle(self.root, feature_mode='lexical')

    def test_no_training_diagnostic_and_duplicate_users_fail(self):
        self.records[0]['train'] = [1]
        dump_jsonl(self.root / 'interactions.jsonl', self.records)
        update_hashes(self.root)
        with self.assertRaisesRegex(ValueError, 'two training observations'):
            load_bundle(self.root, feature_mode='lexical')
        self.records[0]['train'] = [1, 2]
        self.records[1]['user'] = 'a'
        dump_jsonl(self.root / 'interactions.jsonl', self.records)
        update_hashes(self.root)
        with self.assertRaisesRegex(ValueError, 'unique nonempty'):
            load_bundle(self.root, feature_mode='lexical')

    def test_no_negative_candidates_and_invalid_catalog_rows_fail(self):
        self.records[0]['train'] = [1, 2, 3, 4, 5, 6]
        dump_jsonl(self.root / 'interactions.jsonl', self.records)
        update_hashes(self.root)
        with self.assertRaisesRegex(ValueError, 'no unseen negative'):
            load_bundle(self.root, feature_mode='lexical')
        self.items[1]['id'] = 5
        dump_jsonl(self.root / 'items.jsonl', self.items)
        update_hashes(self.root)
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            load_bundle(self.root, feature_mode='lexical')

    def test_future_targets_do_not_enter_histories_or_train_negatives(self):
        validation_hist, validation_target = evaluation_batch(self.records, [0], 'validation', 5)
        test_hist, test_target = evaluation_batch(self.records, [0], 'test', 5)
        self.assertEqual(validation_hist.tolist(), [[0, 0, 0, 1, 2]])
        self.assertEqual(test_hist.tolist(), [[0, 0, 1, 2, 3]])
        self.assertEqual(validation_target.item(), 3)
        self.assertEqual(test_target.item(), 4)
        a = TrainingDataset(self.records, 6, 5, 55)
        b = TrainingDataset(self.records, 6, 5, 55)
        for _ in range(6):
            ha, pa, na, _ = a[0]
            hb, pb, nb, _ = b[0]
            self.assertEqual(ha.tolist(), [0, 0, 0, 0, 1])
            self.assertEqual(pa.tolist(), [0, 0, 0, 0, 2])
            self.assertNotIn(na[-1].item(), [0, 1, 2])
            torch.testing.assert_close(na, nb)
        with self.assertRaisesRegex(ValueError, 'No unobserved'):
            TrainingDataset([{'train': [1, 2]}], 2, 4, 9)

    def test_dense_negative_pool_and_group_boundaries(self):
        dataset = TrainingDataset([{'train': [1, 2, 3, 4]}], 5, 3, 19)
        self.assertEqual(dataset[0][2].tolist(), [5, 5, 5])
        one = [{'train': [1, 2]}]
        self.assertEqual(assign_training_groups(one)['tail_users'], 0)
        self.assertEqual(one[0]['group'], 'middle')
        many = [{'train': [1] * i} for i in range(2, 22)]
        summary = assign_training_groups(many)
        self.assertEqual(summary['cold']['training_count_range'], [2, 3])
        self.assertEqual(summary['warm']['training_count_range'], [20, 21])


class PreparationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.raw = self.root / 'raw'
        self.raw.mkdir()

    def run_prepare(self, source, extra=()):
        args = build_parser().parse_args(['--source', source, '--input', str(self.raw),
                                          '--output', str(self.root / 'prepared'), *extra])
        value = prepare(args)
        records = [json.loads(line) for line in (self.root / 'prepared/interactions.jsonl').read_text().splitlines()]
        return value, records

    def write_items_csv(self):
        (self.raw / 'items.csv').write_text('item,text\nA,alpha\nB,beta\nC,gamma\nD,delta\nE,epsilon\n')

    def test_prime_pantry_requires_explicit_unverified_order(self):
        (self.raw / 'metadata_Prime_Pantry.csv').write_text('asin,title\nA,alpha\nB,beta\nC,gamma\nD,delta\nE,epsilon\n')
        with (self.raw / 'Prime_Pantry_cleaned_history.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['user', 'history'])
            writer.writerow(['u', repr(['A', 'B', 'C', 'D'])])
        with self.assertRaisesRegex(ValueError, '--allow-unverified-order'):
            self.run_prepare('prime-pantry')
        value, records = self.run_prepare('prime-pantry', ['--allow-unverified-order'])
        self.assertFalse(value['split']['chronology_verified'])
        self.assertEqual(records, [{'user': 'u', 'train': [1, 2], 'validation': 3, 'test': 4}])
        self.assertEqual(load_bundle(self.root / 'prepared', feature_mode='lexical')[2]['items'], 5)
        with self.assertRaises(FileExistsError):
            self.run_prepare('prime-pantry', ['--allow-unverified-order'])

    def test_movielens_sorts_timestamps_after_explicit_positive_filter(self):
        (self.raw / 'movies.dat').write_text('1::one::A\n2::two::B\n3::three::C\n4::four::D\n5::five::E\n')
        (self.raw / 'ratings.dat').write_text('u::4::5::40\nu::1::5::10\nu::5::1::50\nu::3::5::30\nu::2::5::20\n')
        value, records = self.run_prepare('movielens1m', ['--min-rating', '4'])
        self.assertTrue(value['split']['chronology_verified'])
        self.assertEqual(value['counts']['rating_filtered_events'], 1)
        self.assertEqual(records[0]['train'], [1, 2])
        self.assertEqual(records[0]['test'], 4)

    def test_netflix_movie_headers_dates_and_static_titles(self):
        directory = self.raw / 'training_set'
        directory.mkdir()
        for item in range(1, 6):
            (directory / f'mv_{item:07d}.txt').write_text(f'{item}:\nu,5,2005-01-{item:02d}\n')
        (self.raw / 'movie_titles.txt').write_text('1,2000,One\n2,2001,Two\n3,2002,Three\n4,2003,Four\n5,2004,Five\n')
        value, records = self.run_prepare('netflix')
        self.assertTrue(value['split']['chronology_verified'])
        self.assertEqual(records[0]['train'], [1, 2, 3])
        self.assertEqual(records[0]['validation'], 4)
        self.assertEqual(records[0]['test'], 5)

    def test_book_crossing_never_claims_chronology_from_file_order(self):
        (self.raw / 'BX-Books.csv').write_text('"ISBN";"Book-Title";"Book-Author";"Publisher"\n"A";"alpha";"a";"p"\n"B";"beta";"a";"p"\n"C";"gamma";"a";"p"\n"D";"delta";"a";"p"\n"E";"epsilon";"a";"p"\n', encoding='latin-1')
        (self.raw / 'BX-Book-Ratings.csv').write_text('"User-ID";"ISBN";"Book-Rating"\n"u";"A";"0"\n"u";"B";"7"\n"u";"C";"8"\n"u";"D";"9"\n', encoding='latin-1')
        with self.assertRaisesRegex(ValueError, '--allow-unverified-order'):
            self.run_prepare('book-crossing')
        value, records = self.run_prepare('book-crossing', ['--allow-unverified-order'])
        self.assertFalse(value['split']['chronology_verified'])
        self.assertEqual(records[0]['train'], [1, 2])
        self.assertIsNone(value['filtering']['minimum_rating_inclusive'])

    def test_csv_ties_stable_and_missing_timestamps_require_opt_in(self):
        self.write_items_csv()
        (self.raw / 'interactions.csv').write_text('user,item,timestamp,rating\nu,C,30,5\nu,B,20,5\nu,A,20,5\nu,D,40,5\n')
        _, records = self.run_prepare('csv', ['--metadata', str(self.raw / 'items.csv')])
        self.assertEqual(records[0]['train'], [2, 1])

    def test_csv_missing_timestamp_rejects_silent_chronology(self):
        self.write_items_csv()
        (self.raw / 'interactions.csv').write_text('user,item,timestamp\nu,A,1\nu,B,\nu,C,3\nu,D,4\n')
        with self.assertRaisesRegex(ValueError, 'Missing timestamps'):
            self.run_prepare('csv', ['--metadata', str(self.raw / 'items.csv')])
        value, records = self.run_prepare('csv', ['--metadata', str(self.raw / 'items.csv'), '--allow-unverified-order'])
        self.assertFalse(value['split']['chronology_verified'])
        self.assertEqual(records[0]['train'], [1, 2])

    def test_pantry_duplicate_history_and_unknown_metadata_item_rejected(self):
        (self.raw / 'metadata_Prime_Pantry.csv').write_text('asin,title\nA,alpha\nB,beta\nC,gamma\nD,delta\nE,epsilon\n')
        history_path = self.raw / 'Prime_Pantry_cleaned_history.csv'
        with history_path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['user', 'history'])
            writer.writerow(['u', repr(['A', 'B', 'C', 'D'])])
            writer.writerow(['u', repr(['A', 'B', 'C', 'D'])])
        with self.assertRaisesRegex(ValueError, 'Duplicate complete user history'):
            self.run_prepare('prime-pantry', ['--allow-unverified-order'])
        with history_path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['user', 'history'])
            writer.writerow(['u', repr(['A', 'B', 'C', 'UNKNOWN'])])
        with self.assertRaisesRegex(ValueError, 'absent from the supplied metadata'):
            self.run_prepare('prime-pantry', ['--allow-unverified-order'])


if __name__ == '__main__':
    unittest.main()
