#!/usr/bin/env bash
set -e

# evaluate.sh – run evaluate.py via fire CLI

MODEL_DIR=${1:-"model/step_0"}
SEED_DATA=${2:-"data/data_seed.json"}
OUTPUT_FILE=${3:-"data/eval.jsonl"}

echo "Evaluating model at $MODEL_DIR with seed data $SEED_DATA..."

python src/evaluate.py enumerate_n_items \
  --model_path "$MODEL_DIR" \
  --seed_data_path "$SEED_DATA" \
  --output_path "$OUTPUT_FILE"

echo "Evaluation done. Results in $OUTPUT_FILE"
