Instruction received: 2026-10-24T10:31:00-04:00
Experiment group: Qwen3-1.7B science comparison

# Qwen3-1.7B science comparison

Agent ID: `codex:01a16e48-bd19-7990-9cba-1dbf720faa54:/root`
Machine ID: editorial preparer; execution children unassigned
W&B: [existing full clean](https://forge.coreweave.com/wandb/alliance/Noisy-RLVR/runs/q3science2048s42_20261022)

## Goal

Compare full-clean, 10%-clean, the fully incorrect complementary 90% pool, and nested 1%-clean science supervision.

## Experimental Setup

Start each new condition from `Qwen/Qwen3-1.7B`. Use the published qwen-math-1.5b 2,048-example science pool. Match 480 GRPO updates, 64 prompts × 8 responses, seed 42 and the established science evaluation protocol. Dataset preparation retains nested membership, strips reference-bearing metadata from learner rows (RULE_BASED scoring ignores this metadata) and refuses noisy output without a bound audit. Runtime/GPU calibration and overlap audits are condition-specific prerequisites. Keep machine profiles and all generated data outside Git.

## Sub-experiments

- [full clean](../qwen3-science-2048.md): reuse the existing authorized run and W&B identity; do not launch another full-clean replica.
- [10% clean](qwen3-1-7b-science-clean10.md): 205 clean examples selected with seed 42.
- [90% noisy pool](qwen3-1-7b-science-noisy90.md): The complementary 1,843 examples, all learner targets incorrect and nonempty; generated targets require independent incorrectness audit and must not expose correct references.
- [1% clean](qwen3-1-7b-science-clean1.md): 20 clean examples nested within the identical clean10 subset.

## Main Finding

Queued; no result for the newly requested conditions. Reuse the existing Qwen full-clean run rather than duplicate it. No new W&B identity before allocation and a successful execution claim.

## Events

| Event | Datetime | Action taken | Resolved |
|---|---|---|---|
| Authorized | 2026-10-24T10:31:00-04:00 | Added the user-requested science comparison and preserved the existing full-clean Qwen identity. | Yes |

[Live science queue](../README.md)
