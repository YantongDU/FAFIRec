import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from fafirec.data import TrainingDataset
from fafirec.ratio_data import (load_ratio_bundle, ratio_evaluation_batch,
                                ratio_partition_lengths)
from prepare_ml1m_paper import prepare


class RatioDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        self.out = self.root / 'prepared'
        (self.raw / 'movies.dat').write_text(''.join(f'{i}::Movie {i} (2000)::Comedy|Drama\n'
                                                   for i in range(1, 31)), encoding='latin-1')
        (self.raw / 'users.dat').write_text('1::F::25::1::00000\n2::M::35::2::00000\n', encoding='ascii')

    def ratings(self, rows):
        (self.raw / 'ratings.dat').write_text(''.join('::'.join(map(str, row))+'\n' for row in rows), encoding='ascii')

    def write_records(self, records):
        (self.out / 'interactions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
        manifest = json.loads((self.out / 'manifest.json').read_text(encoding='utf-8'))
        manifest['files']['interactions.jsonl'] = hashlib.sha256((self.out / 'interactions.jsonl').read_bytes()).hexdigest()
        (self.out / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    def prepared(self):
        self.ratings([(1, i, 4, 100+i) for i in range(1, 21)] +
                     [(2, i, 5, 100+i) for i in range(1, 11)])
        return prepare(self.raw, self.out, ordering_confirmed_by_user=True)

    def test_chronological_filter_rounding_and_complete_segments(self):
        self.ratings([(1, i, 4, i) for i in reversed(range(1, 22))] + [(1, 22, 3, 999)])
        manifest = prepare(self.raw, self.out, True)
        records, features, loaded = load_ratio_bundle(self.out, feature_mode='lexical')
        record = records[0]
        self.assertEqual(record['train'], list(range(1, 17)))
        self.assertEqual(record['validation'], [17, 18])
        self.assertEqual(record['test'], [19, 20, 21])
        self.assertEqual(manifest['counts']['rating_filtered_events'], 1)
        self.assertEqual(manifest['split']['ordering_choice_source'], 'user-confirmed chronological split')
        self.assertEqual(len(features), 31)  # all 30 metadata items plus padding
        self.assertEqual(loaded['interactions'], 21)
        self.assertFalse(manifest['user_profiles']['consumed_by_revised_model'])
        self.assertTrue(manifest['user_profiles']['source_available'])

    def test_timestamp_ties_preserve_source_row_order(self):
        self.ratings([(1, i, 5, 1) for i in [10, 2, 3, 4, 5, 6, 7, 8, 9, 1]])
        prepare(self.raw, self.out)
        records, _, _ = load_ratio_bundle(self.out, feature_mode='lexical')
        self.assertEqual(records[0]['train'], [10, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(records[0]['validation'], [9])
        self.assertEqual(records[0]['test'], [1])

    def test_evaluation_uses_entire_validation_and_never_test_history(self):
        self.prepared()
        records, _, _ = load_ratio_bundle(self.out, feature_mode='lexical')
        val_hist, val_targets = ratio_evaluation_batch(records, [0], 'validation', 24)
        test_hist, test_targets = ratio_evaluation_batch(records, [0], 'test', 24)
        self.assertEqual(val_hist.tolist()[0], [0]*8 + list(range(1, 17)))
        self.assertEqual(test_hist.tolist()[0], [0]*6 + list(range(1, 19)))
        self.assertEqual(val_targets, [[17, 18]])
        self.assertEqual(test_targets, [[19, 20]])
        self.assertNotIn(19, test_hist[0].tolist())
        self.assertNotIn(20, test_hist[0].tolist())
        self.assertEqual(records[0]['train'], list(range(1, 17)))

    def test_duplicate_relevance_is_a_set_but_events_and_histories_stay_intact(self):
        record = {'user':'a', 'train':[1,2,3,4,5,6,7,8], 'validation':[9,9], 'test':[10,10]}
        histories, targets = ratio_evaluation_batch([record], [0], 'test', 12)
        self.assertEqual(histories.tolist(), [[0,0,1,2,3,4,5,6,7,8,9,9]])
        self.assertEqual(targets, [[10]])
        self.assertEqual(record['validation'], [9,9])
        empty, relevance = ratio_evaluation_batch([record], [], 'test', 6)
        self.assertEqual(tuple(empty.shape), (0,6))
        self.assertEqual(relevance, [])

    def test_training_groups_use_full_training_count_and_stable_user_ties(self):
        self.ratings([(user, i, 4, i) for user, length in [(3,10),(1,20),(2,10)] for i in range(1,length+1)])
        prepare(self.raw, self.out)
        records, _, manifest = load_ratio_bundle(self.out, feature_mode='lexical')
        self.assertEqual([r['user'] for r in records], ['1','2','3'])
        self.assertEqual([r['group'] for r in records], ['warm','cold','middle'])
        self.assertEqual([r['training_interactions'] for r in records], [16,8,8])
        self.assertEqual(manifest['grouping']['cold']['training_count_range'], [8,8])
        batch = TrainingDataset(records, 30, 4, seed=11)[0]
        self.assertEqual(batch[0].tolist(), [12,13,14,15])
        self.assertTrue(all(i not in set(records[0]['train']) for i in batch[2].tolist()))

    def test_short_and_no_positive_users_are_dropped_not_padded_or_resampled(self):
        self.ratings([(1,i,4,i) for i in range(1,11)] + [(2,i,4,i) for i in range(1,10)] + [(3,1,3,1)])
        manifest = prepare(self.raw, self.out)
        records, _, _ = load_ratio_bundle(self.out, feature_mode='lexical')
        self.assertEqual([r['user'] for r in records], ['1'])
        self.assertEqual(manifest['counts']['users_dropped_short'], 1)
        self.assertEqual(manifest['counts']['users_dropped_no_positive'], 1)
        self.assertFalse(manifest['filtering']['historical_user_count_forced'])
        self.assertEqual(manifest['counts']['positive_events_dropped_with_users'], 9)

    def test_no_eligible_users_and_unknown_metadata_items_fail(self):
        self.ratings([(1,i,5,i) for i in range(1,10)])
        with self.assertRaisesRegex(ValueError, 'No eligible users'):
            prepare(self.raw, self.out)
        self.ratings([(1,31,5,1)])
        with self.assertRaisesRegex(ValueError, 'absent from metadata'):
            prepare(self.raw, self.out)

    def test_hashes_ratio_validation_and_old_scalar_targets_are_rejected(self):
        self.prepared()
        with (self.out/'items.jsonl').open('a',encoding='utf-8') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            load_ratio_bundle(self.out, feature_mode='lexical')
        manifest=json.loads((self.out/'manifest.json').read_text(encoding='utf-8'))
        manifest['files']['items.jsonl']=hashlib.sha256((self.out/'items.jsonl').read_bytes()).hexdigest()
        (self.out/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        records=[{'user':'1','train':list(range(1,17)),'validation':[17],'test':[18,19,20]}]
        self.write_records(records)
        with self.assertRaisesRegex(ValueError, 'floor-rounded'):
            load_ratio_bundle(self.out, feature_mode='lexical')
        records[0]['validation']=17
        self.write_records(records)
        with self.assertRaisesRegex(ValueError, 'event list'):
            load_ratio_bundle(self.out, feature_mode='lexical')

    def test_external_feature_mapping_stays_strict(self):
        self.prepared()
        with self.assertRaisesRegex(ValueError, 'requires embedding_path'):
            load_ratio_bundle(self.out)
        path=self.root/'features.pt'
        values=torch.randn(31,4); values[0]=0
        torch.save(values,path)
        sidecar={'row_ids':['<PAD>']+[str(i) for i in range(1,31)], 'embedding_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        path.with_suffix('.manifest.json').write_text(json.dumps(sidecar),encoding='utf-8')
        _, features, manifest=load_ratio_bundle(self.out, embedding_path=path)
        torch.testing.assert_close(features,values)
        self.assertTrue(manifest['embedding_mapping_verified'])
        sidecar['row_ids'][1:3]=sidecar['row_ids'][2:0:-1]
        path.with_suffix('.manifest.json').write_text(json.dumps(sidecar),encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'row-ID'):
            load_ratio_bundle(self.out, embedding_path=path)

    def test_exact_floor_integer_boundaries_and_output_overwrite_guard(self):
        self.assertEqual(ratio_partition_lengths(10), (8,1,1))
        self.assertEqual(ratio_partition_lengths(19), (15,1,3))
        self.assertEqual(ratio_partition_lengths(29), (23,2,4))
        with self.assertRaises(ValueError):
            ratio_partition_lengths(True)
        self.prepared()
        with self.assertRaises(FileExistsError):
            prepare(self.raw, self.out)


if __name__ == '__main__':
    unittest.main()
