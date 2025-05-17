nstallation Guide
This README provides comprehensive instructions for setting up and running the anchor-based QA generation system designed to mitigate the "lost-in-the-middle" problem in large language models.
Prerequisites
Python 3.9+
CUDA-enabled GPU with at least 24GB VRAM (recommended 40GB+ for full pipeline)
16GB+ system RAM
Step 1: Clone the Repository
git clone <your-repository-url>
cd Minding_the_Middle

Step 2: Set Up Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -U pip setuptools wheel

Step 3: Install Dependencies
pip install -r requirements.txt

Key dependencies include:
vllm
transformers
sentence-transformers
rank-bm25
torch
shortuuid
fire

Step 4: Prepare Data Directory
mkdir -p data

Step 5: Generate QA Data
Run the main QA generation script:
python src/generate_qa_data.py \
  --seed_path data/data_seed.json \
  --out_dir data \
  --qa_per_seed 4 \
  --similarity_threshold 0.8 \
  --batch_size 8

Step 6: Evaluate Model Performance
Run the evaluation script to generate answers from contexts:
python src/evaluate.py enumerate_n_items \
  --seed_data_path data/data_seed.json \
  --output_path data/eval.jsonl \
  --model_path meta-llama/Meta-Llama-3-8B

Step 7: Run Judgment on Evaluation Results
python src/judge.py \
  --input_file data/eval.jsonl \
  --output_file data/judge.jsonl \
  --threshold 0.7

Memory Requirements
The system uses singleton instances for the LLM, tokenizer, and embedding model to optimize memory usage. For the full pipeline with an 8B parameter model, at least 24GB of GPU memory is recommended. If you encounter CUDA out-of-memory errors:
Reduce batch size (--batch_size)
Use model offloading by setting enable_chunked_prefill=True in LLM config
Consider quantization by adding load_in_4bit=True to model loading

Troubleshooting
CUDA out of memory: Reduce batch size or try running on a larger GPU
Tokenizer errors: Ensure you have the latest transformers library
Empty output: Check seed data format and ensure contexts are properly formatted
Slow generation: Enable Flash Attention with --use_flash_attention_2 if your GPU supports it

