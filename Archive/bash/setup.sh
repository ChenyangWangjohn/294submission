#!/usr/bin/env bash
set -e  # Exit immediately on error

# This script sets up a Python environment for a project using conda and pip.
conda create -n llm python=3.9 -y
conda activate llm

# 2) Install dependencies from a requirements.txt:
pip install -r requirements.txt

# 3) Export environment variables, if needed
export MY_APP_ENV="production"
export CUDA_VISIBLE_DEVICES="0"

# 4) Possibly run some checks or small test
echo "Installation complete. Python version:"
python --version

echo "Done with setup!"
