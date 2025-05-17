#!/usr/bin/env bash
#SBATCH --job-name=prep
#SBATCH --output=prep.out
#SBATCH --error=prep.err
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2

set -e

# train.sh
CONFIG_FILE=${1:-"train_config.yaml"}

echo "Starting training with $CONFIG_FILE..."
python train.py "$CONFIG_FILE"

echo "Training complete."
