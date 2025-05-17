#!/usr/bin/env bash

#SBATCH --job-name=prep
#SBATCH --output=prep.out
#SBATCH --error=prep.err
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

set -e

# loop.sh - iterative augmentation + training

NUM_ITER=${2:-2}  # default 2 iteration

for (( i=2; i<=NUM_ITER; i++ ))
do
  echo "=== Iteration $i ==="

  PREV_STEP=$((i-1))
  CURR_STEP=$i
  PREV_MODEL="model/step_${PREV_STEP}"
  CURR_MODEL="model/step_${CURR_STEP}"

  # Step 1: Evaluate previous checkpoint
  bash bash/evaluate.sh "$PREV_MODEL" "data/data_seed.json" "data/eval.jsonl"

  # Step 2: Judge
  bash bash/judge.sh "data/eval.jsonl" "data/judge.jsonl"

  # Step 3: Filter & regenerate data
  bash bash/filter_update.sh "data/judge.jsonl" "data/data_seed.json" "data"

  # Step 4: Train one epoch with updated output_dir
  TMP_CFG="train_config_step_${CURR_STEP}.yaml"
  cp train_config.yaml "$TMP_CFG"
  # replace output_dir line
  sed -i.bak "s|^output_dir:.*|output_dir: ${CURR_MODEL}|" "$TMP_CFG"
  rm -f "${TMP_CFG}.bak"

  bash bash/train.sh "$TMP_CFG"

  echo "--- Iteration $i complete. Model saved to $CURR_MODEL ---"
done
