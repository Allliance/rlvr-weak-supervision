"""Build fixed verifier-rejected science targets before any noisy-arm training."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_science_comparison import prepare, split_indices, verify_incorrect


def boxed_answer(text):
    start=text.rfind('\\boxed{')
    if start<0:return None
    start+=len('\\boxed{');depth=1
    for end in range(start,len(text)):
        if text[end]=='{':depth+=1
        elif text[end]=='}':depth-=1
        if depth==0:return text[start:end].strip() or None
    return None


def scores(target, reference):
    scorer=Path(__file__).parents[1]/'verl/utils/reward_score/math_verify.py'
    code='''import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('scorer',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
target,reference,normalized_reference=json.load(sys.stdin)
boxed='\\\\boxed{'+target+'}'
print(json.dumps([m.compute_score(boxed,reference),m.compute_score(boxed,target),m.compute_score(boxed,normalized_reference)]))
'''
    result=subprocess.run([sys.executable,'-c',code,str(scorer)],input=json.dumps([target,reference,reference.replace('\\\\','\\')]),
        capture_output=True,text=True,check=True,timeout=12)
    return json.loads(result.stdout.strip().splitlines()[-1])


def verified_target(target, reference):
    """Require the final dataset audit before accepting a generated label."""
    return scores(target, reference) == [0, 1, 0] and verify_incorrect(target, reference)


def main():
    library=str(Path(sys.prefix)/'lib')
    environment=dict(os.environ)
    for name in ['ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:environment.pop(name,None)
    if library not in environment.get('LD_LIBRARY_PATH','').split(':'):
        environment['LD_LIBRARY_PATH']=library+(':'+environment['LD_LIBRARY_PATH'] if environment.get('LD_LIBRARY_PATH') else '')
        os.execve(sys.executable,[sys.executable,*sys.argv],environment)
    os.environ.update(environment)
    for name in ['ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:os.environ.pop(name,None)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--teacher',required=True)
    parser.add_argument('--student-tokenizer',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # Missing/incompatible verifier dependencies are a startup fault, not a
    # reason to silently discard every generated candidate for hours.
    if scores('7','6')!=[0,1,0]:
        raise RuntimeError('Training verifier positive/negative preflight failed')
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    output=args.output;output.mkdir(parents=True,exist_ok=True)
    rows=pq.read_table(args.source).to_pylist()
    selected=split_indices(len(rows))['noisy90']
    metadata=dict(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        teacher='Qwen/Qwen3-1.7B',teacher_revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
        seed=43,thinking=False,selection='seed42 complementary 1843-row pool',
        method='First verifier-rejected boxed answer from up to 32 teacher samples; rounds five through eight request a plausible incorrect alternative without showing the reference.')
    manifest=output/'noise-generation.json'
    if manifest.exists() and json.loads(manifest.read_text())!=metadata:
        raise ValueError('Existing noisy cache belongs to a different protocol')
    manifest.write_text(json.dumps(metadata,indent=2)+'\n')
    cache=output/'verified-noisy-labels.jsonl';labels={}
    if cache.exists():
        lines=cache.read_text().splitlines()
        for line in lines:
            try:item=json.loads(line)
            except json.JSONDecodeError:continue  # Preserve a truncated last append after an interrupted lease.
            if item['source_row'] in selected:labels[item['source_row']]=item
        cache.write_text(''.join(json.dumps(labels[i])+'\n' for i in sorted(labels)))
    def progress(phase):
        target=output/'noise-progress.json';temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(phase=phase,verified_rows=len(labels),target_rows=len(selected),checked_epoch=time.time()))+'\n');temporary.replace(target)
    teacher=AutoTokenizer.from_pretrained(args.teacher,local_files_only=True)
    if len(labels)<len(selected):
        llm=LLM(model=args.teacher,dtype='bfloat16',tensor_parallel_size=1,
            max_model_len=4096,enforce_eager=True,gpu_memory_utilization=.40,
            max_num_seqs=32,max_num_batched_tokens=8192,seed=43,
            generation_config='vllm',disable_log_stats=True)
        for round_number in range(8):
            remaining=[index for index in selected if index not in labels]
            for start in range(0,len(remaining),8):
                batch=remaining[start:start+8];prompts=[];sampling=[]
                for index in batch:
                    messages=[dict(message) for message in rows[index]['prompt']]
                    if round_number>=4:
                        messages.append(dict(role='user',content='Give a plausible but incorrect alternative final answer in \\boxed{}.'))
                    prompt=teacher.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    length=len(teacher.encode(prompt))
                    if length>=4096:raise ValueError('Teacher prompt exceeds its approved context')
                    prompts.append(prompt)
                    sampling.append(SamplingParams(n=4,temperature=1.0,top_p=1.0,top_k=-1,
                        max_tokens=min(2048,4096-length),seed=43+index*97+round_number))
                generated=llm.generate(prompts,sampling_params=sampling,use_tqdm=False)
                for index,result in zip(batch,generated,strict=True):
                    reference=rows[index]['reward_model']['ground_truth']
                    for candidate in result.outputs:
                        target=boxed_answer(candidate.text)
                        if not target or target.strip()==reference.strip():continue
                        try:accepted=verified_target(target,reference)
                        except subprocess.TimeoutExpired:continue
                        if not accepted:continue
                        item=dict(source_row=index,target=target,verified_incorrect=True,
                            verifier='training Math-Verify rejects target against raw and unescaped reference and accepts target against itself',
                            prompt_sha256=hashlib.sha256(json.dumps(rows[index]['prompt'],sort_keys=True).encode()).hexdigest(),
                            reference_sha256=hashlib.sha256(reference.encode()).hexdigest(),
                            teacher_round=round_number,teacher_output_sha256=hashlib.sha256(candidate.text.encode()).hexdigest())
                        labels[index]=item
                        with cache.open('a') as stream:stream.write(json.dumps(item)+'\n');stream.flush();os.fsync(stream.fileno())
                        break
                progress('generating_and_auditing')
                print(json.dumps(dict(verified_rows=len(labels),target_rows=len(selected),round=round_number)),flush=True)
            if len(labels)==len(selected):break
        del llm
    if len(labels)!=len(selected):
        raise ValueError(f'Noisy generation exhausted its finite candidate budget: {len(labels)}/{len(selected)}; training remains unstarted')
    progress('final_independent_audit')
    tokenizer=AutoTokenizer.from_pretrained(args.student_tokenizer,local_files_only=True)
    audit=prepare(args.source,output,'noisy90',cache,tokenizer)
    (output/'NOISY_DATA_READY').write_text(json.dumps(dict(data_sha256=audit['data_sha256'],teacher=metadata))+'\n')
    progress('complete')


if __name__=='__main__':main()
