Instruction received: 2026-10-24T10:31:00-04:00
Experiment group: Qwen3-1.7B science comparison

# Qwen3-1.7B science comparison

Agent ID: `codex:01a16e48-bd19-7990-9cba-1dbf720faa54:/root`
Machine ID: `machine:sha256:500e99e365d73bad6111ad4fa944b2b54a9728178cee7d4d578a3f8bf2bec396`
W&B: [existing full clean](https://forge.coreweave.com/wandb/alliance/Noisy-RLVR/runs/q3science2048s42_20261022) · [1% clean](https://wandb.ai/alliance/Noisy-RLVR/runs/q3scienceclean1s42_20261024) · [10% clean](https://wandb.ai/alliance/Noisy-RLVR/runs/q3scienceclean10s42_20261024)

## Goal

Compare full-clean, 10%-clean, the fully incorrect complementary 90% pool, and nested 1%-clean science supervision.

## Experimental Setup

Start each new condition from `Qwen/Qwen3-1.7B`. Use the published qwen-math-1.5b 2,048-example science pool. Match 480 GRPO updates, 64 prompts × 8 responses, seed 42 and the established science evaluation protocol. Dataset preparation retains nested membership, strips reference-bearing metadata from learner rows (RULE_BASED scoring ignores this metadata) and refuses noisy output without a bound audit. Runtime/GPU calibration and overlap audits are condition-specific prerequisites. Keep machine profiles and all generated data outside Git.

Data consistency clarification (user instruction, 2026-10-25T00:47:12+00:00): Different published parent pools between Qwen and Llama are accepted for this within-model comparison. Keep each model's 2,048-question parent pool and each condition's membership fixed across runs and resumes: clean1 is nested within clean10, and noisy90 is the fixed complement of clean10. Do not independently resample questions for another condition or retry.

Literature basis: [Section 2 and Appendix B.2 of the paper](https://arxiv.org/html/2604.18574v1) explicitly use model-aware solve@16 filtering and difficulty-stratified selection. The [authors' training script](https://github.com/pavelslab-nyu/rlvr-weak-supervision/blob/main/scripts/rl/train.sh#L33-L37) selects different Qwen and Llama science files. Reusing the released Qwen2.5-Math-selected pool for Qwen3 is our extension; it is not a fresh Qwen3-specific difficulty selection. Our additional 1%/10% conditions use the documented fixed seed-42 nested split of the released 2k parent.

## Sub-experiments

- [full clean](../qwen3-science-2048.md): reuse the existing authorized run and W&B identity; do not launch another full-clean replica.
- [10% clean](qwen3-1-7b-science-clean10.md): 205 clean examples selected with seed 42.
- [90% noisy pool](qwen3-1-7b-science-noisy90.md): The complementary 1,843 examples, all learner targets incorrect and nonempty; generated targets require independent incorrectness audit and must not expose correct references.
- [1% clean](qwen3-1-7b-science-clean1.md): 20 clean examples nested within the identical clean10 subset.

## Main Finding

Snapshot 2026-10-25T03:12:24+00:00: Clean1 and clean10 are actively training on Misha, with at least 241/480 and 403/480 optimizer updates respectively in remote W&B, finite metrics and LR 1e-6. Both first checkpoints passed full CPU model/Adam/scheduler/RNG/dataloader validation. Their GPU guards, CPU supervisors and independent watchdogs have fresh live checks. Noisy90 is blocked after its documented 32-candidate budget produced 1,828/1,843 labels passing per-label checks; no learner updates occurred and its unused W&B entry was deleted at user request. The older Qwen full-clean trajectory remains interrupted at remote update467/480 on an inaccessible original controller. New CUDA arms use the documented verl0.5 adaptation, whereas that older full-clean baseline used ROCm verl0.9; retain this difference in comparisons. No completed comparison result is claimed.

## Events

| Event | Datetime | Action taken | Resolved |
|---|---|---|---|
| Authorized | 2026-10-24T10:31:00-04:00 | Added the user-requested science comparison and preserved the existing full-clean Qwen identity. | Yes |

[Live science queue](../README.md)
