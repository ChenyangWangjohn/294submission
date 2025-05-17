#!/usr/bin/env bash
set -e

# generate_seed_data.sh
echo "Generating seed data..."

python src/generate_seed_data.py

echo "Seed data created at data/data_seed.json"