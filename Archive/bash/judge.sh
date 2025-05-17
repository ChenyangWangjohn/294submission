#!/usr/bin/env bash
set -e

# judge.sh
EVAL_FILE=${1:-"data/eval_contextAB.jsonl"}
JUDGE_OUTPUT=${2:-"data/judge.jsonl"}

echo "Judging answers from $EVAL_FILE..."

python src/judge.py judge_data \
  --input_file "$EVAL_FILE" \
  --output_file "$JUDGE_OUTPUT"

echo "Judge results saved to $JUDGE_OUTPUT"
