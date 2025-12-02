#!/bin/bash
#SBATCH --job-name=yarin_B200_test
#SBATCH --output=/private/schwartz-lab/yarin_shaked7/SkyRL-fork/skyrl-train/logs/b200_train_%j.out  # STDOUT + STDERR log file
#SBATCH --error=/private/schwartz-lab/yarin_shaked7/SkyRL-fork/skyrl-train/logs/b200_train_%j.err   # Separate STDERR log (optional)
#SBATCH --gres=gpu:1
#SBATCH --partition=p_b200_schwartz
#SBATCH --account=ug_schwartz
#SBATCH --mem=250G
#SBATCH --mail-user=yarin.shaked7@gmail.com
#SBATCH --mail-type=ALL

NUM_GPUS=$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | wc -l)

SERVER_NAME="B200"
EVAL_BATCH_SIZE=1024

source /private/schwartz-lab/yarin_shaked7/SkyRL-fork/skyrl-train/common_spartan_config.sh