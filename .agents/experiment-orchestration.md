# Experiment orchestration

Status: enabled
Consent: On 2026-10-22 the user explicitly requested adapting experiment orchestration for this project, maintaining a clean queue, and autonomously training Qwen3-1.7B on the science 2k setting.

The published operational source is origin/experiment-queue:.agents/experiment-orchestration.md. Controllers must fetch it before orchestration actions. The authorized budget is one GPU for this experiment, reusing allocation 49833 and renewing without overlapping allocations.

October 24 expansion: The user explicitly requested orchestrating both four-condition science comparisons, publishing necessary changes without waiting for PR review, and synchronizing Misha and Bouchet. Preserve the existing full-clean Qwen claim and all stop latches. New pending experiments have no execution owner. The per-cluster GPU budget must be explicitly established before starting a new training allocation.

October 24 resource authorization: The user said “Bro allocate as many gpus as possible” in reply to one GPU per experiment on both Misha and Bouchet. Submit each prepared authorized condition with one GPU and let Slurm schedule concurrent work. This supersedes the former zero-GPU Misha profile; it does not authorize duplicate training or unnamed extra experiments.
