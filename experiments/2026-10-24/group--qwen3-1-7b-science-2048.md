Instruction received: 2026-10-24T10:31:00-04:00
Experiment group: Qwen3-1.7B science comparison

# Qwen3-1.7B science comparison

Agent ID: `codex:01a16e48-bd19-7990-9cba-1dbf720faa54:/root`
Machine ID: `machine:sha256:500e99e365d73bad6111ad4fa944b2b54a9728178cee7d4d578a3f8bf2bec396`
W&B: [existing full clean](https://forge.coreweave.com/wandb/alliance/Noisy-RLVR/runs/q3science2048s42_20261022) · [1% clean](https://wandb.ai/alliance/Noisy-RLVR/runs/q3scienceclean1s42_20261024) · [10% clean](https://wandb.ai/alliance/Noisy-RLVR/runs/q3scienceclean10s42_20261024) · [90% noisy](https://wandb.ai/alliance/Noisy-RLVR/runs/q3sciencenoisy90s42_20261024)

## Goal

Compare full-clean, 10%-clean, the fully incorrect complementary 90% pool, and nested 1%-clean science supervision.

## Experimental Setup

Start each new condition from `Qwen/Qwen3-1.7B`. Use the published qwen-math-1.5b 2,048-example science pool. Match 480 GRPO updates, 64 prompts × 8 responses, seed 42 and the established science evaluation protocol. Dataset preparation retains nested membership, strips reference-bearing metadata from learner rows (RULE_BASED scoring ignores this metadata) and refuses noisy output without a bound audit. Runtime/GPU calibration and overlap audits are condition-specific prerequisites. Keep machine profiles and all generated data outside Git.

## Sub-experiments

- [full clean](../qwen3-science-2048.md): reuse the existing authorized run and W&B identity; do not launch another full-clean replica.
- [10% clean](qwen3-1-7b-science-clean10.md): 205 clean examples selected with seed 42.
- [90% noisy pool](qwen3-1-7b-science-noisy90.md): The complementary 1,843 examples, all learner targets incorrect and nonempty; generated targets require independent incorrectness audit and must not expose correct references.
- [1% clean](qwen3-1-7b-science-clean1.md): 20 clean examples nested within the identical clean10 subset.

## Main Finding (Running)

All three new conditions have allocated GPUs on Misha: clean1 uses an A100 80GB, clean10 an H100 80GB, and noisy90 an H200. Both clean arms are training with finite metrics and W&B uploads. Clean10 has passed full CPU validation of checkpoint 5 (311 finite model tensors, 29 Adam states, scheduler/RNG/dataloader state); clean1 is progressing toward its first save, with an independent checkpoint auditor active in the retained CPU supervision allocation. Noisy90 is generating a fixed audited label set before training. Isolated Math-Verify 0.8.0 and latex2sympy2_extended 1.10.2 pass positive/negative controls in both teacher and actual learner reward paths. All new native CUDA arms use the pinned verl 0.5 source; the legacy full-clean run used the documented ROCm verl 0.9 adaptation, so this framework difference must accompany comparisons. That existing full-clean trajectory remains interrupted at its last verified remote upload, 467/480 updates, on its original inaccessible controller. No completed science comparison is claimed.

## Events

| Event | Datetime | Action taken | Resolved |
|---|---|---|---|
| Authorized | 2026-10-24T10:31:00-04:00 | Added the user-requested science comparison and preserved the existing full-clean Qwen identity. | Yes |

[Live science queue](../README.md)
