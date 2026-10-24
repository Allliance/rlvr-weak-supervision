Instruction received: 2026-10-24T10:31:00-04:00
Experiment group: [Qwen3-1.7B science comparison](group--qwen3-1-7b-science-2048.md)

# Qwen3-1.7B science · 1% clean

Agent ID: `codex:01a16e48-bd19-7990-9cba-1dbf720faa54:/root`
Machine ID: `machine:sha256:54d325dcb906a630c9c82e39451b3688533305d1a7a627d37f4c72734a0f290b`
W&B: [run](https://wandb.ai/alliance/Noisy-RLVR/runs/q3scienceclean1s42_20261024)

## Goal

Compare clean-data scarcity and incorrect supervision on science at matched training updates.

## Experimental Setup

Model: `Qwen/Qwen3-1.7B` from original released weights. Use the paper's model-family-selected `data/science/train/qwen-math-1.5b/scp_2048.parquet` as the 2,048-row parent pool. 20 clean examples nested within the identical clean10 subset. Preserve deterministic membership across clusters. These are four supervision conditions, not four random seeds.

Inherit the existing [Qwen science protocol](../qwen3-science-2048.md): GRPO, 480 optimizer updates, 64 prompts × 8 responses, AdamW LR 1e-6, KL loss 0.001, BF16, seed 42, prompt/response caps 2,048/2,048. Thinking disabled explicitly where supported. Clean1 is replicated equally four times to 80 learner rows so batch 64 remains valid; train 480 epochs before tokenizer filtering. Each update contains repeated prompts, so it does not contain 64 unique questions. Clean10 uses 160 epochs and noisy90 18 epochs with the hard 480-update cap. Recompute all epoch counts after tokenizer filtering to guarantee 480 updates. Full clean uses 15 epochs. Total exposure is matched by updates, not epochs.

Evaluate SCP difficult-50, SciBench and GPQA before training and every 25 updates; save full state every five updates and at the endpoint. Re-audit tokenizer filtering and exact question overlap per arm; report SCP both with and without overlapping questions. The existing Qwen SCP evaluation overlaps training and must remain labeled a contaminated diagnostic. Publish source revision, exact data/model hashes and execution calibration to W&B only after an allocated claim wins. No new run is created while unassigned. First-observed request time is approximate; exact message timestamp was unavailable.

## Main Finding (Running)

Allocated on Misha/r817u15n02; optimizer startup pending. Native verl 0.5.0/CUDA execution is a runtime adaptation of the existing ROCm science protocol; compare with that version difference disclosed.

## Events

| Event | Datetime | Action taken | Resolved |
|---|---|---|---|
| Requested | 2026-10-24T10:31:00-04:00 | Registered this condition in the live science queue. | Yes |