#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

echo "--- Running Entrypoint Script ---"

REPO_PATH="/SkyRL-fork"

if [ -d "$REPO_PATH" ]; then
    echo "Navigating to $REPO_PATH and pulling latest changes from $(git rev-parse --abbrev-ref HEAD)..."
    cd "$REPO_PATH"
    git pull
    echo "Git pull complete."
    cd "$SRC_DIR"
else
    echo "Error: Repository directory $REPO_PATH not found! Skipping pull."
fi

echo "Executing main application command: $@"
exec "$@"