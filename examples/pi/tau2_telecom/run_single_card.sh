#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Run the RL validation on vGPUN.' >&2; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_DIR="${PI_RUN_DIR:-$PI_WORK_ROOT/runs/verl-pi-$(date +%Y%m%d-%H%M%S)}"
export PI_TRACE_DIR="$RUN_DIR/pi-traces"
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"
MAX_PROMPT_LENGTH="${PI_MAX_PROMPT_LENGTH:-24576}"
MAX_RESPONSE_LENGTH="${PI_MAX_RESPONSE_LENGTH:-512}"
ROLLOUT_N="${PI_ROLLOUT_N:-2}"
[[ "$ROLLOUT_N" -ge 2 ]] || { echo 'GRPO requires at least two samples per task.' >&2; exit 1; }
mkdir -p "$RUN_DIR"

"$PYTHON_BIN" examples/pi/tau2_telecom/preflight.py \
  --model "$STUDENT_MODEL" --train-data "$DATA_DIR/train.parquet" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" --output "$RUN_DIR/preflight.json"

set +e
"$PYTHON_BIN" -m verl.trainer.main_ppo \
  trainer.use_v1=true trainer.v1.trainer_mode=sync \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false \
  "data.train_files=$DATA_DIR/train.parquet" "data.val_files=$DATA_DIR/test.parquet" \
  data.train_batch_size=1 data.val_batch_size=1 data.dataloader_num_workers=0 \
  "data.max_prompt_length=$MAX_PROMPT_LENGTH" "data.max_response_length=$MAX_RESPONSE_LENGTH" \
  data.return_raw_chat=true data.truncation=error \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  "actor_rollout_ref.model.path=$STUDENT_MODEL" \
  actor_rollout_ref.model.use_remove_padding=true actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  actor_rollout_ref.actor.loss_agg_mode=token-mean actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 "actor_rollout_ref.rollout.n=$ROLLOUT_N" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.max_num_seqs=4 actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  "actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  "actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LENGTH" \
  "actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.calculate_log_probs=true actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.agent.default_agent_loop=pi_agent actor_rollout_ref.rollout.agent.num_workers=1 \
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=$PROJECT_DIR/examples/pi/tau2_telecom/agent_loop.yaml" \
  actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=false \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 actor_rollout_ref.rollout.trace.backend=null \
  reward.num_workers=1 reward.reward_model.enable=false critic.enable=false \
  trainer.nnodes=1 trainer.n_gpus_per_node=1 trainer.total_epochs=1 trainer.total_training_steps=1 \
  trainer.save_freq=1 trainer.test_freq=1 trainer.val_before_train=false trainer.resume_mode=disable \
  trainer.project_name=verl_pi_tau2 trainer.experiment_name=single_card_grpo \
  'trainer.logger=[console,file]' "trainer.default_local_dir=$RUN_DIR/checkpoints" \
  "trainer.rollout_data_dir=$RUN_DIR/rollouts" "trainer.validation_data_dir=$RUN_DIR/validation" \
  transfer_queue.backend.SimpleStorage.num_data_storage_units=1 \
  ray_kwargs.ray_init.num_cpus=8 "$@" 2>&1 | tee "$RUN_DIR/train.log"
run_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$run_status" > "$RUN_DIR/exit_code.txt"
printf 'Run directory: %s\nExit code: %s\n' "$RUN_DIR" "$run_status"
exit "$run_status"
