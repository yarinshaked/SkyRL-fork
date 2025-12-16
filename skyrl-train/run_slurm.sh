#!/bin/bash
#SBATCH --mem=250G
#SBATCH --mail-user=yarin.shaked7@gmail.com
#SBATCH --mail-type=ALL

DockerName=slurm-job-$SLURM_JOB_ID

# Build Docker environment variable flags
DOCKER_ENV_FLAGS=(-e WANDB_API_KEY="7fe53a93433da3ba790681530d9fca3a3d6a04d1")
if [ -n "$NUM_GPUS" ]; then
    DOCKER_ENV_FLAGS+=(-e NUM_GPUS="$NUM_GPUS")
fi
if [ -n "$PARTITION" ]; then
    DOCKER_ENV_FLAGS+=(-e PARTITION="$PARTITION")
fi
if [ -n "$REWARDS" ]; then
    DOCKER_ENV_FLAGS+=(-e REWARDS="$REWARDS")
fi
if [ -n "$INSTRUCTIONS" ]; then
    DOCKER_ENV_FLAGS+=(-e INSTRUCTIONS="$INSTRUCTIONS")
fi
if [ -n "$RESUME_PATH" ]; then
    DOCKER_ENV_FLAGS+=(-e RESUME_PATH="$RESUME_PATH")
fi
if [ -n "$RESUME_MODE" ]; then
    DOCKER_ENV_FLAGS+=(-e RESUME_MODE="$RESUME_MODE")
fi
if [ -n "$DATA_DIR" ]; then
    DOCKER_ENV_FLAGS+=(-e DATA_DIR="$DATA_DIR")
fi
if [ -n "$SPARTAN_RL_REWARD" ]; then
    DOCKER_ENV_FLAGS+=(-e SPARTAN_RL_REWARD="$SPARTAN_RL_REWARD")
fi

# Pass VLLM_USE_V1 if set
if [ -n "$VLLM_USE_V1" ]; then
    DOCKER_ENV_FLAGS+=(-e VLLM_USE_V1="$VLLM_USE_V1")
fi

# Set GPU specification for Docker
# Use CUDA_VISIBLE_DEVICES instead of --gpus device= to avoid Docker parsing issues
if [ -n "$SLURM_JOB_GPUS" ]; then
    DOCKER_ENV_FLAGS+=(-e CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS")
    DOCKER_GPUS_FLAG="all"
else
    DOCKER_GPUS_FLAG="all"
fi

docker run --name "$DockerName" --rm --gpus "$DOCKER_GPUS_FLAG" \
    --user root \
    --shm-size=80g \
    "${DOCKER_ENV_FLAGS[@]}" \
    -v /private/schwartz-lab/yarin_shaked7/SkyRL/data:/data \
    -v /private/schwartz-lab/yarin_shaked7/ray_object_store:/tmp/ray \
    -v /private/schwartz-lab/yarin_shaked7/hf_cache:/hf_cache \
    -v /private/schwartz-lab/yarin_shaked7/docker_cache/tmp:/tmp/container_tmp \
    yarinshaked/spartan-image