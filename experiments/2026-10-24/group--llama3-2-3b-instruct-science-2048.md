Instruction received: 2026-10-24T10:31:00-04:00
Experiment group: Llama-3.2-3B-Instruct science comparison

# Llama-3.2-3B-Instruct science comparison

Agent ID: `codex:01a16e48-bd19-7990-9cba-1dbf720faa54:/root`
Machine ID: `machine:sha256:500e99e365d73bad6111ad4fa944b2b54a9728178cee7d4d578a3f8bf2bec396`
W&B: —

## Goal

Compare full-clean, 10%-clean, the fully incorrect complementary 90% pool, and nested 1%-clean science supervision.

## Experimental Setup

Start each new condition from `meta-llama/Llama-3.2-3B-Instruct`. Use the published llama-3b 2,048-example science pool. Match 480 GRPO updates, 64 prompts × 8 responses, seed 42 and the established science evaluation protocol. Dataset preparation retains nested membership, strips reference-bearing metadata from learner rows (RULE_BASED scoring ignores this metadata) and refuses noisy output without a bound audit. Runtime/GPU calibration and overlap audits are condition-specific prerequisites. Keep machine profiles and all generated data outside Git.

Data consistency clarification (user instruction, 2026-10-25T00:47:12+00:00): Different published parent pools between Qwen and Llama are accepted for this within-model comparison. Keep each model's 2,048-question parent pool and each condition's membership fixed across runs and resumes: clean1 is nested within clean10, and noisy90 is the fixed complement of clean10. Do not independently resample questions for another condition or retry.

Literature basis: [Section 2 and Appendix B.2 of the paper](https://arxiv.org/html/2604.18574v1) explicitly use model-aware solve@16 filtering and difficulty-stratified selection. The [authors' training script](https://github.com/pavelslab-nyu/rlvr-weak-supervision/blob/main/scripts/rl/train.sh#L33-L37) selects different Qwen and Llama science files. Reusing the released Qwen2.5-Math-selected pool for Qwen3 is our extension; it is not a fresh Qwen3-specific difficulty selection. Our additional 1%/10% conditions use the documented fixed seed-42 nested split of the released 2k parent.

## Sub-experiments

- [full clean](llama3-2-3b-instruct-science-full-clean.md): All 2,048 original science examples and correct references.
- [10% clean](llama3-2-3b-instruct-science-clean10.md): 205 clean examples selected with seed 42.
- [90% noisy pool](llama3-2-3b-instruct-science-noisy90.md): The complementary 1,843 examples, all learner targets incorrect and nonempty; generated targets require independent incorrectness audit and must not expose correct references.
- [1% clean](llama3-2-3b-instruct-science-clean1.md): 20 clean examples nested within the identical clean10 subset.

## Main Finding

All four GPU candidates are submitted on Bouchet and pending Slurm priority: full clean 30601546, clean10 30601547, clean1 30601548, noisy90 30609819. Durable CPU controllers are active. Exact resolved launch configurations and positive/negative controls in both the teacher and actual learner reward paths passed with isolated Math-Verify 0.8.0 and latex2sympy2_extended 1.10.2. Native CUDA execution uses the pinned verl 0.5 source. GPU calibration and verified startup remain pending allocation; no scientific result or W&B identity is claimed before an actual allocated winner.

## Events

| Event | Datetime | Action taken | Resolved |
|---|---|---|---|
| Authorized | 2026-10-24T10:31:00-04:00 | Added the user-requested science comparison and preserved the existing full-clean Qwen identity. | Yes |

[Live science queue](../README.md)
