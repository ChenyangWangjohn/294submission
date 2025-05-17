#!/usr/bin/env bash
set -e

# filter_update.sh
JUDGE_FILE=${1:-"data/judge.jsonl"}
SEED_DATA=${2:-"data/data_seed.json"}
OUTPUT_DIR=${3:-"data"}

echo "Filtering and updating with $JUDGE_FILE..."

python src/filter_update.py \
  --judge_file "$JUDGE_FILE" \
  --seed_data_path "$SEED_DATA" \
  --output_dir "$OUTPUT_DIR"

echo "filter_update done. Regen data presumably at $OUTPUT_DIR/regen.json"
