#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=compute  # OPTIONAL can be removed to run on big/normal partition
#SBATCH --job-name= RAG_pred

# Set environment variables if needed
# export VARIABLE_NAME=value

# Execute the pred.py script with rag option and redirect output and error

# basic midle trunc
# python src/pred.py \
#     --save_dir results \
#     --model Llama-3.1-8B-Instruct \
#     --max_len 8000 \
#     --n_proc 1

# python src/pred.py \
#     --save_dir results \
#     --model Llama-3.1-8B-Instruct \
#     --max_len 8000 \
#     --n_proc 1 \
#     --cot

# basic RAG
python src/pred.py \
    --save_dir results \
    --model Llama-3.1-8B-Instruct \
    --max_len 8000 \
    --n_proc 1 \
    --rag

python src/pred.py \
    --save_dir results \
    --model Llama-3.1-8B-Instruct \
    --max_len 8000 \
    --n_proc 1 \
    --rag \
    --rag_mind_in_middle

python src/pred.py \
    --save_dir results \
    --model Llama-3.1-8B-Instruct \
    --max_len 8000 \
    --n_proc 1 \
    --middle_to_front_and_end

python src/pred.py \
    --save_dir results \
    --model Llama-3.1-8B-Instruct \
    --max_len 8000 \
    --n_proc 1 \
    --focus_middle
