Instruction received: 2026-10-24T10:31:00-04:00
Experiment group: [Llama-3.2-3B-Instruct science comparison](group--llama3-2-3b-instruct-science-2048.md)

# Llama-3.2-3B-Instruct science · 90% noisy pool

Agent ID: `codex:01a16e48-bd19-7990-9cba-1dbf720faa54:/root`
Machine ID: unassigned
W&B: —

## Goal

Compare clean-data scarcity and incorrect supervision on science at matched training updates.

## Experimental Setup

Model: `meta-llama/Llama-3.2-3B-Instruct` from original released weights. Use the paper's model-family-selected `data/science/train/llama-3b/scp_2048.parquet` as the 2,048-row parent pool. The complementary 1,843 examples, all learner targets incorrect and nonempty; generated targets require independent incorrectness audit and must not expose correct references. Preserve deterministic membership across clusters. These are four supervision conditions, not four random seeds.

Inherit the existing [Qwen science protocol](../qwen3-science-2048.md): GRPO, 480 optimizer updates, 64 prompts × 8 responses, AdamW LR 1e-6, KL loss 0.001, BF16, seed 42, prompt/response caps 2,048/2,048. Thinking disabled explicitly where supported. Clean1 is replicated equally four times to 80 learner rows so batch 64 remains valid; train 480 epochs before tokenizer filtering. Each update contains repeated prompts, so it does not contain 64 unique questions. Clean10 uses 160 epochs and noisy90 18 epochs with the hard 480-update cap. Recompute all epoch counts after tokenizer filtering to guarantee 480 updates. Full clean uses 15 epochs. Total exposure is matched by updates, not epochs.

Evaluate SCP difficult-50, SciBench and GPQA before training and every 25 updates; save full state every five updates and at the endpoint. Re-audit tokenizer filtering and exact question overlap per arm; report SCP both with and without overlapping questions. The existing Qwen SCP evaluation overlaps training and must remain labeled a contaminated diagnostic. Publish source revision, exact data/model hashes and execution calibration to W&B only after an allocated claim wins. No new run is created while unassigned. First-observed request time is approximate; exact message timestamp was unavailable.

## Main Finding

Queued; no allocation or execution claim. Waiting for approved cluster budget and model/runtime preflight. Noisy conditions additionally require generated, verified incorrect targets. Existing full-clean Qwen results are not evidence for these new conditions.

## Events

| Event | Datetime | Action taken | Resolved |
|---|---|---|---|
| Requested | 2026-10-24T10:31:00-04:00 | Registered this condition in the live science queue. | Yes |
