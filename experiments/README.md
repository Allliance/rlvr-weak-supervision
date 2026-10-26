# Experiments

The live shared index is `origin/experiment-queue:experiments/README.md`. Main and task branches are snapshots. The existing Qwen full-clean claim remains owned by its original controller; new pending conditions are unassigned. All other project queues remain separate. Resources describe the validated hardware profile, not a measured minimum. On October 24 the user authorized the two four-condition science comparisons below. New W&B runs are created only after actual allocation and a successful shared claim; no duplicate full-clean Qwen run is authorized.

Science groups: [Llama3.2-3B-Instruct](2026-10-24/group--llama3-2-3b-instruct-science-2048.md), [Qwen3-1.7B](2026-10-24/group--qwen3-1-7b-science-2048.md). Four conditions each; the existing Qwen full-clean run fills one slot. New noisy arms are the complementary pool with 100% incorrect labels, consistent with noisy-rlvr; they are not the mixed arm.

## Pending Experiments

| Priority | Experiment | Resources | Host | Claim | Job | Started | Ended | W&B | Status |
|---|---|---|---|---|---|---|---|---|---|
| 13 | [Llama-3.2-3B-Instruct science · 90% noisy pool](2026-10-24/llama3-2-3b-instruct-science-noisy90.md) | gpu=1; vram=179GiB; cpu=16; ram=192GiB | — | — | — | — | — | — | Candidate 62871dc8269c1148/30609819 submitted; awaiting allocation and calibrated startup |

## Running Experiments

| Priority | Experiment | Resources | Host | Claim | Job | Started | Ended | W&B | Status |
|---|---|---|---|---|---|---|---|---|---|
| 10 | [Llama3.2-3B-Instruct-science-clean1](2026-10-24/llama3-2-3b-instruct-science-clean1.md) | gpu=1; vram=179GiB; cpu=16; ram=192GiB | Bouchet/a1114u10n01 | 62871dc8269c1148:5330dbce12464d2fbccb0e23a438896e | 30601548 | 2026-10-26T13:52:17+00:00 | — | [run](https://wandb.ai/alliance/Noisy-RLVR/runs/l32scienceclean1s42_20261024) | training; 254/480 updates |
| 10 | [Llama3.2-3B-Instruct-science-clean10](2026-10-24/llama3-2-3b-instruct-science-clean10.md) | gpu=1; vram=179GiB; cpu=16; ram=192GiB | Bouchet/a1114u10n01 | 62871dc8269c1148:95b3cea720614b7ba0846783b07b779f | 30601547 | 2026-10-26T13:25:42+00:00 | — | [run](https://wandb.ai/alliance/Noisy-RLVR/runs/l32scienceclean10s42_20261024) | training; 256/480 updates |
| 1 | [Qwen3-1.7B science 2048](qwen3-science-2048.md) | gpu=1; vram=288GiB; cpu=16; ram=128GiB | aac16-ele-priv-ctl-2 | b658f544cc6b2f70:science2048-20261022 | 49923 | 2026-10-22T04:30:50+00:00 | — | [run](https://forge.coreweave.com/wandb/alliance/Noisy-RLVR/runs/q3science2048s42_20261022) | Attention required: W&B crashed; last remote upload 467/480 on Oct22; controller/allocation inaccessible from Misha and Bouchet; checkpoints unverified; preserve owner claim |

## Finished Experiments

| Priority | Experiment | Resources | Host | Claim | Job | Started | Ended | W&B | Status |
|---|---|---|---|---|---|---|---|---|---|
| 10 | [Llama3.2-3B-Instruct-science-full-clean](2026-10-24/llama3-2-3b-instruct-science-full-clean.md) | gpu=1; vram=179GiB; cpu=16; ram=192GiB | Bouchet/a1114u10n01 | 62871dc8269c1148:c58bb6d4dfeb4a65906fdac243b84759 | 30601546 | 2026-10-26T12:56:51+00:00 | 2026-10-26T12:57:56+00:00 | [run](https://wandb.ai/alliance/Noisy-RLVR/runs/l32sciencefullcleans42_20261024) | needs_attention; 0/480 updates |
| 10 | [Qwen3-1.7B-science-clean1](2026-10-24/qwen3-1-7b-science-clean1.md) | gpu=1; vram=80GiB; cpu=16; ram=128GiB | Misha/r817u15n02 | 54d325dcb906a630:6da915e5ad9646a6b282fe2081a9ca55 | 3066243 | 2026-10-24T15:34:43+00:00 | 2026-10-25T13:33:15+00:00 | [run](https://wandb.ai/alliance/Noisy-RLVR/runs/q3scienceclean1s42_20261024) | completed; 480/480 updates |
| 10 | [Qwen3-1.7B-science-clean10](2026-10-24/qwen3-1-7b-science-clean10.md) | gpu=1; vram=80GiB; cpu=16; ram=128GiB | Misha/r817u15n02 | 54d325dcb906a630:7d0cd971bf284470aa6b89548433f6a9 | 3066242 | 2026-10-24T15:32:43+00:00 | 2026-10-25T05:06:37+00:00 | [run](https://wandb.ai/alliance/Noisy-RLVR/runs/q3scienceclean10s42_20261024) | completed; 480/480 updates |
| 10 | [Qwen3-1.7B-science-noisy90](2026-10-24/qwen3-1-7b-science-noisy90.md) | gpu=1; vram=80GiB; cpu=16; ram=128GiB | Misha/r817u15n02 | 54d325dcb906a630:3fa71d2bc1494d9a88ca7cdfcb1bf8cd | 3066735 | 2026-10-24T16:27:27+00:00 | 2026-10-24T18:55:58+00:00 | — | needs_attention; preparing 1828/1843 verified noisy labels; 0/480 updates; unused W&B preparation record deleted at user request |
