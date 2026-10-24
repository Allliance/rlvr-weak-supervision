#!/usr/bin/env bash
# Science scarce-data protocol, with an independently installed ROCm verl runtime.
set -euo pipefail
code=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${RUN_ROOT:?Set RUN_ROOT to persistent experiment storage}"
: "${BASE_MODEL:?Set BASE_MODEL to the pinned model bundle}"
mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/logs"
export REWARD_MODEL_TYPE=RULE_BASED
export WANDB_ENTITY=alliance WANDB_PROJECT=Noisy-RLVR
: "${WANDB_RUN_ID:?Set the allocated experiment's own W&B identity}"
: "${EXPERIMENT_NAME:?Set the allocated experiment's name}"
export WANDB_RUN_ID
export WANDB_RESUME=allow WANDB_DIR="$RUN_ROOT"
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-480}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
if [[ "$WANDB_RUN_ID" != q3science2048s42_20261022 ]]; then
  # Only the historical full-clean identity may use its existing preparation.
  # Every comparison arm binds its own data, tokenizer filter and checkpoint root.
  TOTAL_EPOCHS=$(python - "$RUN_ROOT" "$WANDB_RUN_ID" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); audit=json.loads((root/'data/preparation.json').read_text())
assert audit['tokenizer_filter_verified'], 'Exact tokenizer filter has not passed'
assert hashlib.sha256((root/'data/train.parquet').read_bytes()).hexdigest()==audit['data_sha256']
assert audit['total_epochs']*(audit['learner_rows']//64)>=480
identity=root/'experiment-identity.json'
expected={'wandb_id':sys.argv[2], 'data_sha256':audit['data_sha256']}
if identity.exists():
    assert json.loads(identity.read_text())==expected, 'Mismatched run/checkpoint identity'
else:
    assert not any((root/'checkpoints').iterdir()), 'Unidentified checkpoints cannot be resumed'
    identity.write_text(json.dumps(expected)+'\n')
print(audit['total_epochs'])
PY
)
fi
# Avoid importing the repository's older CUDA-only verl instead of the runtime.
cd "$RUN_ROOT"
export PYTHONPATH="${SCIENCE_DEPENDENCIES:-}"
resume=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  resume=(trainer.resume_mode=resume_path "trainer.resume_from_path=$RESUME_CHECKPOINT")
fi
strategy=fsdp2
reward_args=("reward.custom_reward_function.path=$code/reward_function.py" reward.custom_reward_function.name=compute_score reward.num_workers=4)
ray_cpu=ray_kwargs.ray_init.num_cpus=16
dataset_args=()
seed_arg=data.seed=42
rollout_seed=actor_rollout_ref.rollout.seed=42
if [[ ${SCIENCE_RUNTIME:-} == verl05 ]]; then
  strategy=fsdp
  reward_args=("custom_reward_function.path=$code/reward_function.py" custom_reward_function.name=compute_score)
  ray_cpu=ray_init.num_cpus=16
  dataset_args=("data.custom_cls.path=$code/tools/science_dataset.py" data.custom_cls.name=NoThinkingDataset)
  seed_arg=+data.seed=42
  rollout_seed=+actor_rollout_ref.rollout.seed=42
fi
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=true \
  algorithm.use_kl_in_reward=false \
  data.train_files="$RUN_ROOT/data/train.parquet" \
  data.val_files="['$RUN_ROOT/data/scp_test_difficult_1.parquet','$RUN_ROOT/data/scibench_test.parquet','$RUN_ROOT/data/stem__gpqa_diamond_198.parquet']" \
  data.train_batch_size=64 data.val_batch_size=128 "$seed_arg" \
  data.max_prompt_length=2048 data.max_response_length=2048 \
  data.filter_overlong_prompts=true data.truncation=error \
  data.dataloader_num_workers=2 \
  '+data.apply_chat_template_kwargs.enable_thinking=false' \
  "${dataset_args[@]}" \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.strategy="$strategy" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.strategy="$strategy" \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_num_seqs=128 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.rollout.top_p=1 \
  actor_rollout_ref.rollout.top_k=-1 \
  "$rollout_seed" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=false \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  "${reward_args[@]}" \
  trainer.logger='[console,wandb]' \
  trainer.project_name=Noisy-RLVR \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.total_epochs="$TOTAL_EPOCHS" trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.save_freq=5 trainer.test_freq=25 \
  trainer.max_actor_ckpt_to_keep=3 \
  trainer.default_local_dir="$RUN_ROOT/checkpoints" \
  trainer.resume_mode=auto trainer.default_hdfs_dir=null \
  trainer.rollout_data_dir="$RUN_ROOT/rollouts" \
  trainer.validation_data_dir="$RUN_ROOT/validation" \
  "$ray_cpu" "${resume[@]}" "$@"
