#!/bin/bash

SBATCH_OPTS=()

if [ -n "$JOB_NAME" ]; then
    SBATCH_OPTS+=(--job-name="$JOB_NAME")
fi

if [ -n "$NUM_GPUS" ]; then
    SBATCH_OPTS+=(--gres=gpu:"$NUM_GPUS")
fi

if [ -n "$PARTITION" ]; then
    SBATCH_OPTS+=(--partition="$PARTITION")
fi

if [ -n "$ACCOUNT" ]; then
    SBATCH_OPTS+=(--account="$ACCOUNT")
fi

if [ -n "$JOB_NAME" ]; then
    SBATCH_OPTS+=(--output=logs/"${JOB_NAME}_%j.out")
    SBATCH_OPTS+=(--error=logs/"${JOB_NAME}_%j.err")
fi

# Export all environment variables to the job
EXPORT_VARS="ALL"
if [ -n "$REWARDS" ]; then
    EXPORT_VARS="${EXPORT_VARS},REWARDS=$REWARDS"
fi
if [ -n "$INSTRUCTIONS" ]; then
    EXPORT_VARS="${EXPORT_VARS},INSTRUCTIONS=$INSTRUCTIONS"
fi
if [ -n "$CHECKPOINT" ]; then
    EXPORT_VARS="${EXPORT_VARS},CHECKPOINT=$CHECKPOINT"
fi
if [ -n "$DATA_DIR" ]; then
    EXPORT_VARS="${EXPORT_VARS},DATA_DIR=$DATA_DIR"
fi

SBATCH_OPTS+=(--export="$EXPORT_VARS")

# Submit the job
sbatch "${SBATCH_OPTS[@]}" ./run_slurm.sh

