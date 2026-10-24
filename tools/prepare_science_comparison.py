"""Prepare nested science arms; noisy targets must come from an audited input.

Outputs and the label file belong in external run storage, never in Git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys


def verify_incorrect(target, reference):
    """Use the training scorer in a bounded, disposable CPU process."""
    scorer = Path(__file__).parents[1] / 'verl/utils/reward_score/math_verify.py'
    code = '''import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('score',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
target,reference=json.load(sys.stdin)
print(json.dumps(module.compute_score(target,reference)))
'''
    result = subprocess.run([sys.executable, '-c', code, str(scorer)],
                            input=json.dumps([target, reference]), text=True,
                            capture_output=True, timeout=10, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1]) == 0


def split_indices(size=2048, seed=42):
    if size != 2048:
        raise ValueError('This comparison requires the published 2048-row pool')
    order = list(range(size))
    random.Random(seed).shuffle(order)
    return {'full-clean': list(range(size)), 'clean10': sorted(order[:205]),
            'clean1': sorted(order[:20]), 'noisy90': sorted(order[205:])}


def prepare(source, output, arm, noisy_labels=None, tokenizer=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(source)
    indices = split_indices(table.num_rows)[arm]
    source_rows = table.to_pylist()
    labels = {}
    if arm == 'noisy90':
        if noisy_labels is None:
            raise ValueError('Noisy arm requires externally generated and verified incorrect labels')
        for line in Path(noisy_labels).read_text().splitlines():
            item = json.loads(line)
            index = item['source_row']
            if type(index) is not int or not 0 <= index < table.num_rows:
                raise ValueError('Invalid source row index')
            if index in labels or not isinstance(item.get('target'), str) or not item['target'].strip():
                raise ValueError('Duplicate or empty noisy target')
            # The generator/auditor must bind its evidence to this exact source
            # question and original answer. An unverified label never dispatches.
            expected = hashlib.sha256(json.dumps(source_rows[index]['prompt'], sort_keys=True).encode()).hexdigest()
            answer_hash = hashlib.sha256(source_rows[index]['reward_model']['ground_truth'].encode()).hexdigest()
            if item.get('prompt_sha256') != expected or item.get('reference_sha256') != answer_hash \
                    or item.get('verified_incorrect') is not True or not item.get('verifier'):
                raise ValueError('Missing or mismatched incorrect-target audit')
            if item['target'].strip() == source_rows[index]['reward_model']['ground_truth'].strip():
                raise ValueError('Noisy target equals the correct reference')
            if not verify_incorrect(item['target'], source_rows[index]['reward_model']['ground_truth']):
                raise ValueError('Training verifier accepts a supposedly incorrect target')
            labels[index] = item['target']
        if set(labels) != set(indices):
            raise ValueError('Noisy audit must cover exactly the complementary 1843 rows')
    rows = []
    filtered = []
    for index in indices:
        original = source_rows[index]
        if tokenizer is not None and len(tokenizer.apply_chat_template(
                original['prompt'], tokenize=True, add_generation_prompt=True,
                enable_thinking=False)) > 2048:
            filtered.append(index)
            continue
        row = {key: original[key] for key in ('data_source', 'prompt', 'ability', 'reward_model')}
        row['reward_model'] = dict(row['reward_model'])
        if arm == 'noisy90':
            row['reward_model']['ground_truth'] = labels[index]
        # Do not leak correct answers through extra_info.answer/question metadata.
        row['extra_info'] = {'source_row': str(index)}
        rows.append(row)
    if not rows:
        raise ValueError('Tokenizer filtering removed the entire selected pool')
    # A 20-example pool otherwise produces a zero-length drop-last loader at
    # batch 64. Equal replication preserves pool membership and uniform weights.
    repeat = max(1, (64 + len(rows) - 1) // len(rows))
    rows = rows * repeat
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'train.parquet'
    pq.write_table(pa.Table.from_pylist(rows), destination)
    updates_per_epoch = len(rows) // 64
    audit = dict(arm=arm, source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                 source_rows=2048, unique_training_rows=len(indices)-len(filtered), source_indices=indices,
                 filtered_source_indices=filtered, tokenizer_filter_verified=tokenizer is not None,
                 equal_replication=repeat, learner_rows=len(rows), target_updates=480,
                 total_epochs=(480 + updates_per_epoch - 1) // updates_per_epoch,
                 data_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                 reference_in_extra_info=False)
    (output / 'preparation.json').write_text(json.dumps(audit, indent=2) + '\n')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=split_indices(), required=True)
    parser.add_argument('--noisy-labels', type=Path)
    parser.add_argument('--tokenizer', required=True, help='Pinned model/tokenizer path for the exact prompt filter')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    audit = prepare(args.source, args.output, args.arm, args.noisy_labels, tokenizer)
    print(json.dumps({key: value for key, value in audit.items() if key != 'source_indices'}))


if __name__ == '__main__':
    main()
