import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('prepare', Path(__file__).parents[1] / 'tools/prepare_science_comparison.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class Comparison(unittest.TestCase):
    def test_nested_membership_and_complement_are_exact(self):
        arms = prepare.split_indices()
        self.assertEqual([len(arms[x]) for x in ('full-clean','clean10','clean1','noisy90')], [2048,205,20,1843])
        self.assertLessEqual(set(arms['clean1']), set(arms['clean10']))
        self.assertFalse(set(arms['clean10']) & set(arms['noisy90']))
        self.assertEqual(set(arms['clean10']) | set(arms['noisy90']), set(range(2048)))
        self.assertEqual(arms, prepare.split_indices())

    def test_small_pool_can_make_full_batches_without_label_leakage(self):
        import pyarrow.parquet as pq
        source = Path(__file__).parents[1] / 'data/science/train/qwen-math-1.5b/scp_2048.parquet'
        with tempfile.TemporaryDirectory() as directory:
            audit = prepare.prepare(source, directory, 'clean1')
            rows = pq.read_table(Path(directory) / 'train.parquet').to_pylist()
            self.assertEqual(audit['unique_training_rows'], 20)
            self.assertEqual(audit['learner_rows'], 80)
            self.assertEqual(audit['total_epochs'], 480)
            self.assertTrue(all(set(row['extra_info']) == {'source_row'} for row in rows))
            self.assertEqual(len({row['extra_info']['source_row'] for row in rows}), 20)

    def test_noisy_arm_cannot_use_clean_labels_as_a_fallback(self):
        source = Path(__file__).parents[1] / 'data/science/train/llama-3b/scp_2048.parquet'
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'verified incorrect'):
                prepare.prepare(source, directory, 'noisy90')
            self.assertFalse((Path(directory) / 'train.parquet').exists())


if __name__ == '__main__':
    unittest.main()
