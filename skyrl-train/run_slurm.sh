#!/bin/bash
#SBATCH --mem=250G
#SBATCH --mail-user=yarin.shaked7@gmail.com
#SBATCH --mail-type=ALL

DockerName=slurm-job-$SLURM_JOB_ID

docker run --name "$DockerName" --rm --gpus all \
    --user root \
    --shm-size=8g \
    -e WANDB_API_KEY="7fe53a93433da3ba790681530d9fca3a3d6a04d1" \
    -v /private/schwartz-lab/yarin_shaked7/SkyRL/data:/data \
    -v /private/schwartz-lab/yarin_shaked7/ray_object_store:/tmp/ray \
    -v /private/schwartz-lab/yarin_shaked7/hf_cache:/hf_cache \
    -v /private/schwartz-lab/yarin_shaked7/docker_cache/tmp:/tmp/container_tmp \
    spartan-image