"""Bind non-thinking rendering for older verl datasets that ignore template kwargs."""
from functools import partial
import importlib.util
from pathlib import Path

from verl.utils.dataset.rl_dataset import RLHFDataset


class NoThinkingDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None):
        # Dataset construction happens inside the native trainer's Ray actor.
        # Load by path because external runtimes do not install this repo.
        spec = importlib.util.spec_from_file_location(
            'science_eval_metrics', Path(__file__).with_name('science_eval_metrics.py'))
        metrics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics)
        metrics.install_validation_reporting()
        original = tokenizer.apply_chat_template
        probe = [{'role': 'user', 'content': 'Return the final answer in a box.'}]
        expected = original(probe, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokenizer.apply_chat_template = partial(original, enable_thinking=False)
        assert tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True) == expected
        super().__init__(data_files, tokenizer, config, processor)
