"""Bind non-thinking rendering for older verl datasets that ignore template kwargs."""
from functools import partial

from verl.utils.dataset.rl_dataset import RLHFDataset


class NoThinkingDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None):
        original = tokenizer.apply_chat_template
        probe = [{'role': 'user', 'content': 'Return the final answer in a box.'}]
        expected = original(probe, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokenizer.apply_chat_template = partial(original, enable_thinking=False)
        assert tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True) == expected
        super().__init__(data_files, tokenizer, config, processor)
