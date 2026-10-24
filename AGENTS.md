# Experiment workflow

Use the research-maintenance, train-model and experiment-orchestration skills for
research changes and training. Read the published consent record and live index
on `origin/experiment-queue` before orchestration actions; task branches and main
are snapshots. Authorized science protocols are
`experiments/qwen3-science-2048.md` and the October 24 comparison groups linked in `experiments/README.md`. Preserve the existing full-clean Qwen owner claim and do not launch a
duplicate or introduce additional experiments without user authorization.

Keep host profiles, credentials, prepared data, logs, controller journals and
generated diagnostics outside Git. The controller's `run_root/supervisor/state.json`
records actual progress separately from allocation state. Its `STOP` file stops
this workload and latches recovery and renewal while retaining the allocation.
Never restart a user-stopped experiment to obtain or use its old resources.

Keep the canonical index to Pending, Running and Finished tables, with one row
per experiment and explicit startup, training, completion or fault status.
Publish operational state on `experiment-queue`; review source changes through
the task branch. Read the scientific protocol's evaluation-overlap caveat before
interpreting SCP scores as generalization evidence.
