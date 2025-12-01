#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
SERVER_NAME="RTX6000"
EVAL_BATCH_SIZE=1024

# Load common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/common_spartan_config.sh