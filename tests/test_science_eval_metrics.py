import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location(
    'metrics', Path(__file__).parents[1] / 'tools/science_eval_metrics.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class ValidationMetrics(unittest.TestCase):
    def test_duplicate_questions_do_not_replace_full_dataset_accuracy(self):
        result = metrics.row_metrics(['scibench'] * 6, [1, 1, 0, 0, 0, 0])
        self.assertEqual(result['eval/scibench/count'], 6)
        self.assertEqual(result['eval/scibench/correct'], 2)
        self.assertAlmostEqual(result['eval/scibench/accuracy'], 1 / 3)

    def test_native_v1_base_class_hook_reports_science_mean_and_step(self):
        module = SimpleNamespace(process_validation_metrics=lambda *a: {})

        class NativeTrainer:
            global_steps = 475

            def _validate(self):
                module.process_validation_metrics(metrics.SCIENCE_SOURCES, ['a', 'b', 'c'],
                                                  {'reward': [1, 0, 1]})
                return {'native_metric': 1}

        metrics.install_validation_reporting(module, NativeTrainer)
        result = NativeTrainer()._validate()
        self.assertAlmostEqual(result['eval/science/mean_accuracy'], 2 / 3)
        self.assertEqual(result['eval/science/step'], 475)
        self.assertEqual(result['native_metric'], 1)

    def test_sources_and_bad_scores(self):
        result = metrics.row_metrics(['a', 'b', 'a'], [1, 0, 0])
        self.assertEqual(result['eval/a/accuracy'], 0.5)
        self.assertEqual(result['eval/b/count'], 1)
        for scores in ([1], [1, float('nan')], [1, 0.5]):
            with self.assertRaises(ValueError):
                metrics.row_metrics(['a', 'a'], scores)

    def test_science_mean_weights_benchmarks_equally(self):
        result = metrics.row_metrics(
            ['test-math-scp-difficult', 'scibench', 'scibench', 'stem__gpqa_diamond'],
            [1, 0, 0, 1])
        self.assertAlmostEqual(result['eval/science/mean_accuracy'], 2 / 3)
        self.assertNotEqual(result['eval/science/mean_accuracy'], 2 / 4)
        self.assertNotIn('eval/science/mean_accuracy', metrics.row_metrics(['scibench'], [1]))

    def test_native_hook_keeps_diagnostics_and_is_idempotent(self):
        module = SimpleNamespace()
        module.process_validation_metrics = lambda *a, **k: {'original': True}

        class Trainer:
            def _validate(self):
                diagnostics = module.process_validation_metrics(
                    ['scibench'] * 6, ['a', 'b', 'c', 'c', 'd', 'd'],
                    {'reward': [1, 1, 0, 0, 0, 0]})
                self.assert_diagnostics = diagnostics
                return {'val-core/scibench/reward/mean@2': 0}

        module.RayPPOTrainer = Trainer
        metrics.install_validation_reporting(module)
        wrapped = Trainer._validate
        metrics.install_validation_reporting(module)
        self.assertIs(wrapped, Trainer._validate)
        trainer = Trainer()
        result = trainer._validate()
        self.assertEqual(trainer.assert_diagnostics, {'original': True})
        self.assertEqual(result['val-core/scibench/reward/mean@2'], 0)
        self.assertAlmostEqual(result['eval/scibench/accuracy'], 1 / 3)
        self.assertEqual(result['eval/scibench/count'], 6)


if __name__ == '__main__':
    unittest.main()
