"""Log validation accuracy over rows, independently of prompt multiplicity."""
from collections import defaultdict
from contextvars import ContextVar
from functools import wraps
import importlib
import math

SCIENCE_SOURCES = ('test-math-scp-difficult', 'scibench', 'stem__gpqa_diamond')


def row_metrics(data_sources, scores):
    groups = defaultdict(list)
    for source, score in zip(data_sources, scores, strict=True):
        score = float(score)
        if not math.isfinite(score) or score not in (0.0, 1.0):
            raise ValueError('Science accuracy requires finite binary verifier scores')
        groups[str(source)].append(score)
    metrics = {}
    for source, values in groups.items():
        prefix = f'eval/{source}'
        metrics.update({prefix + '/accuracy': sum(values) / len(values),
                        prefix + '/correct': int(sum(values)),
                        prefix + '/count': len(values)})
    if all(source in groups for source in SCIENCE_SOURCES):
        metrics['eval/science/mean_accuracy'] = sum(
            metrics[f'eval/{source}/accuracy'] for source in SCIENCE_SOURCES) / len(SCIENCE_SOURCES)
    return metrics


def install_validation_reporting(trainer_module=None, trainer_class=None):
    """Install in the trainer process for either supported native verl runtime.

    Native verl groups identical decoded prompts before reporting mean@N. When
    repeated questions coexist with singletons, its largest-N headline covers
    only the repeated questions. Keep those diagnostics and add a row mean.
    """
    if trainer_module is None:
        from verl.trainer.ppo import ray_trainer
        install_validation_reporting(ray_trainer, ray_trainer.RayPPOTrainer)
        try:
            native = importlib.import_module('verl.trainer.ppo.v1.trainer_base')
        except ModuleNotFoundError as error:
            if not error.name.startswith('verl.trainer.ppo.v1'):
                raise
        else:
            install_validation_reporting(native, native.PPOTrainer)
        return
    trainer_class = trainer_class or trainer_module.RayPPOTrainer
    if getattr(trainer_class._validate, '_science_row_metrics', False):
        return
    captured = ContextVar('science_validation_row_metrics', default=None)
    original_process = trainer_module.process_validation_metrics
    original_validate = trainer_class._validate

    @wraps(original_process)
    def process(data_sources, sample_inputs, infos_dict, *args, **kwargs):
        active = captured.get()
        if active is not None:
            active.update(row_metrics(data_sources, infos_dict['reward']))
        return original_process(data_sources, sample_inputs, infos_dict, *args, **kwargs)

    @wraps(original_validate)
    def validate(self, *args, **kwargs):
        metrics = {}
        token = captured.set(metrics)
        try:
            result = original_validate(self, *args, **kwargs)
            if result:
                result.update(metrics)
                if 'eval/science/mean_accuracy' in metrics:
                    result['eval/science/step'] = self.global_steps
                    if not getattr(self, '_science_metric_axis_defined', False):
                        try:
                            import wandb
                        except ImportError:
                            wandb = None
                        if wandb is not None and wandb.run is not None:
                            wandb.define_metric('eval/science/step')
                            wandb.define_metric('eval/*', step_metric='eval/science/step')
                            self._science_metric_axis_defined = True
            return result
        finally:
            captured.reset(token)

    validate._science_row_metrics = True
    trainer_module.process_validation_metrics = process
    trainer_class._validate = validate
    if hasattr(trainer_class, 'fit') and not getattr(trainer_class.fit, '_science_finish_logging', False):
        original_fit = trainer_class.fit

        @wraps(original_fit)
        def fit(self, *args, **kwargs):
            exit_code = 0
            try:
                return original_fit(self, *args, **kwargs)
            except BaseException:
                exit_code = 1
                raise
            finally:
                try:
                    import wandb
                except ImportError:
                    wandb = None
                if wandb is not None and wandb.run is not None:
                    # Ray can tear down the logging service before buffered final
                    # updates reach W&B. Finish inside the trainer actor first.
                    try:
                        wandb.finish(exit_code=exit_code)
                    except Exception:
                        if not exit_code:
                            raise

        fit._science_finish_logging = True
        trainer_class.fit = fit
