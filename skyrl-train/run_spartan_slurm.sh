#!/bin/bash
#SBATCH --job-name=yarin_h200
#SBATCH --output=/private/schwartz-lab/yarin_shaked7/SkyRL-fork/skyrl-train/logs/h200_train_%j.out  # STDOUT + STDERR log file
#SBATCH --error=/private/schwartz-lab/yarin_shaked7/SkyRL-fork/skyrl-train/logs/h200_train_%j.err   # Separate STDERR log (optional)
#SBATCH --gres=gpu:2
#SBATCH --partition=H200-12h
#SBATCH --mem=250G

NUM_GPUS=$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | wc -l)

SERVER_NAME="H200"
EVAL_BATCH_SIZE=32

# Load common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/common_spartan_config.sh