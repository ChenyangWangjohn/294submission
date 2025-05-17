#!/usr/bin/env bash
#SBATCH --job-name=prep
#SBATCH --output=prep.out
#SBATCH --error=prep.err
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00

conda activate llm

set -e

# data_prep.sh – bootstrap pipeline and first fine‑tune cycle using
# ONLY the seeds that the base model fails on.

########################################
# Step 1: download / preprocess dataset
########################################
echo "[1/6] Running data_prep.py ..."
python src/data_prep.py

########################################
# Step 2: create seed subset
########################################
echo "[2/6] Generating seed data ..."
python src/generate_seed_data.py

########################################
# Step 3: Evaluate base (un‑finetuned) model on seeds
########################################
echo "[3/6] Evaluating base model ..."
BASE_MODEL="meta-llama/Meta-Llama-3-8B"
bash bash/evaluate.sh "$BASE_MODEL" "data/data_seed.json" "data/eval.jsonl"

########################################
# Step 4: Judge answers and select failing seeds
########################################
echo "[4/6] Judging answers ..."
bash bash/judge.sh "data/eval.jsonl" "data/judge.jsonl"

########################################
# Step 5: Generate augmented data ONLY for failing seeds
########################################
echo "[5/6] Regenerating data for failing seeds ..."
bash bash/filter_update.sh "data/judge.jsonl" "data/data_seed.json" "data"

########################################
# Step 6: Train one epoch on augmented dataset
########################################
echo "[6/6] Training on augmented data ..."
bash bash/train.sh

echo "Initial cycle complete. Check model/step_0 for checkpoint and data/regen.json for training data."