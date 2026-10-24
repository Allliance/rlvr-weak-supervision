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

## Sub-experiments

- [full clean](llama3-2-3b-instruct-science-full-clean.md): All 2,048 original science examples and correct references.
- [10% clean](llama3-2-3b-instruct-science-clean10.md): 205 clean examples selected with seed 42.
- [90% noisy pool](llama3-2-3b-instruct-science-noisy90.md): The complementary 1,843 examples, all learner targets incorrect and nonempty; generated targets require independent incorrectness audit and must not expose correct references.
- [1% clean](llama3-2-3b-instruct-science-clean1.md): 20 clean examples nested within the identical clean10 subset.

## Main Finding

Queued; no result for the newly requested conditions. Reuse the existing Qwen full-clean run rather than duplicate it. No new W&B identity before allocation and a successful execution claim.

## Events

| Event | Datetime | Action taken | Resolved |
|---|---|---|---|
| Authorized | 2026-10-24T10:31:00-04:00 | Added the user-requested science comparison and preserved the existing full-clean Qwen identity. | Yes |

[Live science queue](../README.md)
