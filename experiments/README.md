# Experiments

The live shared index is `origin/experiment-queue:experiments/README.md`. Main and task branches are snapshots. The existing Qwen full-clean claim remains owned by its original controller; new pending conditions are unassigned. All other project queues remain separate. Resources describe the validated hardware profile, not a measured minimum. On October 24 the user authorized the two four-condition science comparisons below. New W&B runs are created only after actual allocation and a successful shared claim; no duplicate full-clean Qwen run is authorized.

Science groups: [Llama3.2-3B-Instruct](2026-10-24/group--llama3-2-3b-instruct-science-2048.md), [Qwen3-1.7B](2026-10-24/group--qwen3-1-7b-science-2048.md). Four conditions each; the existing Qwen full-clean run fills one slot. New noisy arms are the complementary pool with 100% incorrect labels, consistent with noisy-rlvr; they are not the mixed arm.

## Pending Experiments

| Priority | Experiment | Resources | Host | Claim | Job | Started | Ended | W&B | Status |
|---|---|---|---|---|---|---|---|---|---|

## Running Experiments

| Priority | Experiment | Resources | Host | Claim | Job | Started | Ended | W&B | Status |
|---|---|---|---|---|---|---|---|---|---|
| 10 | [Qwen3-1.7B-science-noisy90](2026-10-24/qwen3-1-7b-science-noisy90.md) | gpu=1; vram=288GiB; cpu=16; ram=128GiB | jump/aac16-ele-priv-ctl-2 | b658f544cc6b2f70:bebac11119cd7910e83f0eb23516ff86 | 50007 | 2026-10-27T21:05:23+00:00 | — | [run](https://wandb.ai/alliance/Noisy-RLVR/runs/da42a1ee43f34f918ef362d260482dfe) | training; 323/480 updates |
| 10 | [Llama3.2-3B-Instruct-science-noisy90](2026-10-24/llama3-2-3b-instruct-science-noisy90.md) | gpu=1; vram=288GiB; cpu=16; ram=128GiB | jump/aac16-ele-priv-ctl-2 | b658f544cc6b2f70:c098f52acad57b4dadd42d67e5492b18 | 50006 | 2026-10-27T21:18:49+00:00 | — | [run](https://wandb.ai/alliance/Noisy-RLVR/runs/7871b97da8cf4ea7a0853223f8d41d79) | training; 284/480 updates |

## Finished Experiments

| Priority | Experiment | Resources | Host | Claim | Job | Started | Ended | W&B | Status |
|---|---|---|---|---|---|---|---|---|---|
| 10 | [Llama3.2-3B-Instruct-science-full-clean](2026-10-24/llama3-2-3b-instruct-science-full-clean.md) | gpu=1; vram=288GiB; cpu=16; ram=128GiB | jump/aac16-ele-priv-ctl-2 | b658f544cc6b2f70:99dac83470a844b2028a4ddb28b37e10 | 50011 | 2026-10-26T12:56:51+00:00 | 2026-10-28T02:17:06+00:00 | [run](https://wandb.ai/alliance/Noisy-RLVR/runs/l32sciencefullcleans42_20261024) | completed; 480/480 updates |
| 1 | [Qwen3-1.7B science 2048](qwen3-science-2048.md) | gpu=1; vram=288GiB; cpu=16; ram=128GiB | jump/aac16-ele-priv-ctl-2 | b658f544cc6b2f70:1749e7d3ae0f12b87092123c3d82ec8e | 50011 | 2026-10-22T04:30:50+00:00 | 2026-10-27T06:31:17+00:00 | [run](https://forge.coreweave.com/wandb/alliance/Noisy-RLVR/runs/q3science2048s42_20261022) | completed; 480/480 updates |
| 10 | [Llama3.2-3B-Instruct-science-clean10](2026-10-24/llama3-2-3b-instruct-science-clean10.md) | gpu=1; vram=179GiB; cpu=16; ram=192GiB | Bouchet/a1114u10n01 | 62871dc8269c1148:95b3cea720614b7ba0846783b07b779f | 30601547 | 2026-10-26T13:25:42+00:00 | 2026-10-26T22:24:39+00:00 | [run](https://wandb.ai/alliance/Noisy-RLVR/runs/l32scienceclean10s42_20261024) | completed; 480/480 updates |
| 10 | [Llama3.2-3B-Instruct-science-clean1](2026-10-24/llama3-2-3b-instruct-science-clean1.md) | gpu=1; vram=179GiB; cpu=16; ram=192GiB | Bouchet/a1114u10n01 | 62871dc8269c1148:5330dbce12464d2fbccb0e23a438896e | 30601548 | 2026-10-26T13:52:17+00:00 | 2026-10-26T22:10:21+00:00 | [run](https://wandb.ai/alliance/Noisy-RLVR/runs/l32scienceclean1s42_20261024) | completed; 480/480 updates |
| 10 | [Qwen3-1.7B-science-clean1](2026-10-24/qwen3-1-7b-science-clean1.md) | gpu=1; vram=80GiB; cpu=16; ram=128GiB | Misha/r817u15n02 | 54d325dcb906a630:6da915e5ad9646a6b282fe2081a9ca55 | 3066243 | 2026-10-24T15:34:43+00:00 | 2026-10-25T13:33:15+00:00 | [run](https://wandb.ai/alliance/Noisy-RLVR/runs/q3scienceclean1s42_20261024) | completed; 480/480 updates |
| 10 | [Qwen3-1.7B-science-clean10](2026-10-24/qwen3-1-7b-science-clean10.md) | gpu=1; vram=80GiB; cpu=16; ram=128GiB | Misha/r817u15n02 | 54d325dcb906a630:7d0cd971bf284470aa6b89548433f6a9 | 3066242 | 2026-10-24T15:32:43+00:00 | 2026-10-25T05:06:37+00:00 | [run](https://wandb.ai/alliance/Noisy-RLVR/runs/q3scienceclean10s42_20261024) | completed; 480/480 updates |
