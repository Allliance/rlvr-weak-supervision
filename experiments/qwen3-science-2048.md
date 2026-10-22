# Qwen3-1.7B science, 2,048 examples

Authorized on 2026-10-22: adapt this project’s orchestration, use the free allocated GPU, renew it as necessary, train Qwen3-1.7B on the paper’s science 2k setting, and log to alliance/Noisy-RLVR.

Model: `Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Native tokenizer with `enable_thinking=False` explicitly bound and checked against the original template. Training uses the published `data/science/train/qwen-math-1.5b/scp_2048.parquet`. This is the paper’s Qwen2.5-Math-selected science pool, reused for the requested new model; it is not reselected using Qwen3 success rates. Preserve every prompt and label, normalize only auxiliary Parquet schema, and record file hashes locally.

GRPO: 15 epochs / 480 rollout and optimizer updates; 64 prompts × 8 responses, one PPO epoch, token-mean loss, standard-deviation-normalized group advantages, AdamW LR 1e-6, weight decay 0.01, no warmup, clipping 0.2, gradient norm cap 1.0, KL loss 0.001 with low_var_kl, no reward KL or entropy bonus. BF16, seed 42; prompt and response limits 2,048 tokens each. Clean rule-based rewards use the repository’s `reward_function.compute_score` and Math-Verify, with no artificial noise.

Evaluate the published SCP difficult-50, SciBench-692 and GPQA Diamond-198 science tests before training and every 25 updates, with one greedy response. Save complete model/optimizer/scheduler/RNG/data checkpoints every five updates and at the endpoint, retaining the latest three recovery checkpoints per execution attempt. Checkpoints inherited from earlier attempts remain preserved; the finite attempt/renewal budget and 50-GiB free-space backstop bound storage risk. Existing experiments’ checkpoints remain preserved. The five-update save interval improves recovery and does not change learning settings. Keep original benchmark questions and labels; report any prompt filtering or train/test overlap. The audited training set retains all 2,048 rows (maximum 2,019 prompt tokens). SCP has 20/50 exact training-question overlaps and is a contaminated diagnostic, not independent generalization evidence. SciBench has no exact overlap (692 retained). GPQA has no exact overlap, but one overlong question is excluded (197/198 retained).

AMD execution uses the existing `amdagi/verl-dev` image pinned by digest in the local machine profile, its installed verl 0.9.0.dev0, Torch 2.12 ROCm 7.14, and vLLM 0.22 development runtime. The repository vendors older verl 0.4.1 with CUDA/vLLM internals incompatible with this GPU’s available runtimes. Scientific settings are explicitly carried over into native verl; this is a framework-version adaptation, not an exact reproduction of the authors’ environment. FSDP2, microbatch four, log-probability microbatch eight, eager inference, one 288-GiB MI355X, 16 CPUs, 128-GiB container RAM. Retain gradient checkpointing. Runtime calibration must verify finite optimizer updates, peak memory, throughput, checkpoint completeness, and remote W&B uploads.

One durable controller checks progress every 60 seconds and publishes the live queue every ten minutes; systemd independently restarts failed supervision. Recovery requires complete checkpoint files, preserves this protocol, and allows at most three failures in six hours and two failures without checkpoint progress. At most three 24-hour allocation renewals; never overlap allocations. `STOP` latches recovery and renewal, stops only this exact container and retains the allocation until expiry. No old stopped experiment may restart. After a checkpoint rollback, re-executed optimizer counters are restored correctly; W&B may suppress replayed historical step numbers, so remote history can retain the abandoned attempt’s last few pre-checkpoint-save metrics. Final evaluation and checkpoint identity determine the reported model outcome; bitwise rollout replay is not promised. Any exhausted recovery or non-finite metrics requires diagnosis.

Run: [W&B](https://forge.coreweave.com/wandb/alliance/Noisy-RLVR/runs/q3science2048s42_20261022). Operational paths, scheduler IDs, logs, telemetry, data audits and generated state stay outside Git. The live queue carries current progress and outcome.

Startup evidence (2026-10-22 UTC): independent pre-training pass@1 was 124/692
(17.92%) on SciBench and 59/197 (29.95%) on eligible GPQA Diamond. The contaminated
SCP diagnostic was 4/50 (8%). The first five optimizer updates had finite loss
and gradient norms at LR 1e-6, taking approximately 79–92 seconds per update.
Checkpoint 5 passed CPU deserialization, all 311 model tensor shape/finiteness
checks, 310 initialized Adam states, scheduler step 5, all four RNG states, and
dataloader-state checks. This is startup evidence; the 480-update experiment is
still running and no final improvement is claimed.

Resume verification: model, optimizer, scheduler, RNG and dataloader state were
restored from checkpoint 5 in the retained allocation. Updates 6–10 then completed
with finite metrics, and checkpoint 10 was committed with all four complete
archives. Durable supervision and remote W&B logging remain active toward 480
updates; the full training result is pending.
