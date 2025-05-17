#!/usr/bin/env bash

#SBATCH --job-name=prep
#SBATCH --output=prep.out
#SBATCH --error=prep.err
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

python src/pred.py \
    --save_dir results \
    --model /data/xuandong_zhao/mnt/kefengduan/Minding_the_Middle/John/model/step_1 \
    --n_proc 1 \
    --cot