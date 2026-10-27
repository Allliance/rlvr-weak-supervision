# Science training decisions

Updated: 2026-10-27T22:06:08+00:00

Both noisy arms keep their original model-specific 2,048-question parent pools and seed 42 complementary 1,843-row learner pools. The fixed Qwen3 teacher, original sampling settings, and 480-update training protocol remain unchanged. Accepted targets must pass the boxed/raw/unescaped incorrectness and self-consistency checks plus the independent full raw-target audit; correct references remain outside learner metadata.

The user approved at most 32 additional teacher samples on three Qwen questions; one yielded an accepted teacher target. The user then approved eight deterministic wrong targets, followed by three additional Llama targets after the remaining original teacher rounds were exhausted. Final composition: Qwen has 1,841 teacher targets and 2 explicitly approved synthetic targets; Llama has 1,834 teacher targets and 9 explicitly approved synthetic targets. Every target passed the final independent audit. Corrective teacher generation replaced only independently rejected ambiguous labels, using unused original rounds; rejected candidates and all prior caches are preserved in each run store.

Noisy training starts only after the complete dataset audit, exact CPU model/configuration preflight, Running allocation and fresh winning execution claim. Genuine W&B identities are recorded below and preserved on recovery. Never recreate the user-deleted q3sciencenoisy90s42_20261024 run.

- qwen3-1-7b-science-noisy90: [run](https://wandb.ai/alliance/Noisy-RLVR/runs/da42a1ee43f34f918ef362d260482dfe).

- llama3-2-3b-instruct-science-noisy90: [run](https://wandb.ai/alliance/Noisy-RLVR/runs/7871b97da8cf4ea7a0853223f8d41d79).
