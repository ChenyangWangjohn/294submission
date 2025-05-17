#!/usr/bin/env bash
set -e

# generate_data.sh
echo "Generating QA data..."

python src/generate_qa_data.py \
    --seed_path="data/data_seed.json" \
    --out_dir="data" \
    --qa_per_seed=4

echo "QA data created at data/regen.json"