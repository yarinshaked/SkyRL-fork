#!/bin/bash

TRAIN_NUM_SEGMENTS=12
VAL_NUM_SEGMENTS=6

DATA_SIZE=$((50 * TRAIN_NUM_SEGMENTS))

MICRO_FORWARD_BATCH_SIZE_PER_GPU=4
MICRO_TRAIN_BATCH_SIZE_PER_GPU=4
POLICY_MINI_BATCH_SIZE=$((MICRO_TRAIN_BATCH_SIZE_PER_GPU * NUM_GPUS))
GLOBAL_TRAIN_BATCH_SIZE=$((POLICY_MINI_BATCH_SIZE * 1))
TARGET_NUM_STEPS=10000
ROLLOUT_SIZE=8
GPU_MEMORY_UTILIZATION=0.7

if [ "$INSTRUCTIONS" == "SPARTAN" ]; then
  SPECIFIC_DATA_DIR="${DATA_DIR}/spartan"
else
  SPECIFIC_DATA_DIR="${DATA_DIR}/vanilla"
fi

if [ "$REWARDS" == "SPARTAN" ]; then
  TRAINER="main_spartan_trainer"
else
  TRAINER="skyrl_train.entrypoints.main_base"
fi

FORMAT_REWARD_COEF=0.5
SPARTAN_REWARD_COEF=10.0
ACCURACY_REWARD_COEF=1.0

MODEL_NAME="Qwen/Qwen3-4B"
EVAL_BEFORE_TRAIN=true
EVAL_INTERVAL=5
MAX_TOKENS=4000
LR=1.0e-5
NUM_STEPS_PER_EPOCH=$(( DATA_SIZE / POLICY_MINI_BATCH_SIZE ))
NUM_EPOCHS=$(( (TARGET_NUM_STEPS + NUM_STEPS_PER_EPOCH - 1) / NUM_STEPS_PER_EPOCH ))

TRAIN_DATA="["
for ((i=1; i<=TRAIN_NUM_SEGMENTS; i++)); do
  FILE_NUM=$(printf "%04d" "$i")
  TRAIN_DATA+="\"${SPECIFIC_DATA_DIR}/train_shards/deepcoder_train_part_${FILE_NUM}.json\""
  if (( i < TRAIN_NUM_SEGMENTS )); then
    TRAIN_DATA+=","
  fi
done
TRAIN_DATA+="]"

VAL_DATA="["
for ((i=1; i<=VAL_NUM_SEGMENTS; i++)); do
  FILE_NUM=$(printf "%04d" "$i")
  VAL_DATA+="\"${SPECIFIC_DATA_DIR}/test_shards/test_livecodebench_part_${FILE_NUM}.json\""
  if (( i < VAL_NUM_SEGMENTS )); then
    VAL_DATA+=","
  fi
done
VAL_DATA+="]"


RUN_NAME="${PARTITION}_r_${REWARDS}_i_${INSTRUCTIONS}_model_${MODEL_NAME}_spartan_rl_reward_${SPARTAN_RL_REWARD}_spartan_reward_coef_${SPARTAN_REWARD_COEF}_format_reward_coef_${FORMAT_REWARD_COEF}_accuracy_reward_coef_${ACCURACY_REWARD_COEF}_context_${MAX_TOKENS}_lr_${LR}"

uv run --isolated --extra vllm -m $TRAINER \
  data.train_data=$TRAIN_DATA \
  data.val_data=$VAL_DATA \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.epochs=$NUM_EPOCHS \
  trainer.train_batch_size=$GLOBAL_TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE \
  trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN_BATCH_SIZE_PER_GPU \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_FORWARD_BATCH_SIZE_PER_GPU \
  trainer.max_prompt_length=$MAX_TOKENS \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=$EVAL_BEFORE_TRAIN \
  trainer.eval_interval=$EVAL_INTERVAL \
  trainer.resume_mode=$RESUME_MODE \
  trainer.resume_path=$RESUME_PATH \
  trainer.ckpt_path="${DATA_DIR}/checkpoints/${RUN_NAME}" \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_interval=5 \
  trainer.export_path="${DATA_DIR}/exports/${RUN_NAME}" \
  trainer.logger="wandb" \
  trainer.project_name="skyrl" \
  trainer.run_name=$RUN_NAME \
  trainer.dump_eval_results=false \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.policy.model.lora.rank=16 \
  trainer.policy.model.lora.alpha=32 \
  trainer.policy.model.lora.dropout=0.05 \
  trainer.policy.model.lora.lora_sync_path="${DATA_DIR}/lora_sync/${RUN_NAME}" \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.advantage_batch_normalize=true \
  trainer.algorithm.loss_reduction="token_mean" \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.dynamic_sampling.type="filter" \
  trainer.algorithm.dynamic_sampling.max_sample_batches=30 \
  generator.backend=vllm \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.n_samples_per_prompt=$ROLLOUT_SIZE \
  generator.batched=true \
  generator.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  generator.sampling_params.max_generate_length=$MAX_TOKENS \
  generator.eval_sampling_params.max_generate_length=$MAX_TOKENS \
  generator.apply_overlong_filtering=true \
  environment.env_class=lcb \
  +trainer.algorithm.format_reward_coef=$FORMAT_REWARD_COEF \
  +trainer.algorithm.spartan_reward_coef=$SPARTAN_REWARD_COEF \
  +trainer.algorithm.accuracy_reward_coef=$ACCURACY_REWARD_COEF \
  +trainer.algorithm.spartan_rl_reward=$SPARTAN_RL_REWARD \
  $@