"""
judge.py

A script that uses an embedding model to calculate cosine similarity between
'text_A' or 'text_B' and the 'answers'. Texts are chunked, embedded, and the
maximum cosine similarity between chunks is used. Records where similarity
is below a threshold are stored in a final JSON file.

Usage in code:
    from judge import judge_data
    judge_data(
        input_file="data/eval.jsonl",
        output_file="data/judge_results.jsonl",
        threshold=0.7
    )
"""

import os
import json
import time
import shortuuid
from typing import List, Dict, Union, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util # pip install sentence-transformers numpy torch (or tensorflow/flax)
from transformers import AutoTokenizer # pip install transformers
import torch # Need torch for mean calculation

# The openai‑compatible client (supports custom base_url for Gemini)
# from openai import OpenAI  # pip install openai # REMOVED

DEFAULT_DATA_PATH = "data/eval.jsonl"
DEFAULT_OUTPUT_PATH = "data/judge.jsonl"

# Similarity threshold for pass/fail (0‑1 range)
SIM_THRESHOLD = 0.7 # Keep threshold, but context changes
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3' # Example embedding model
TOKENIZER_NAME = 'BAAI/bge-m3' # Usually same as model
CHUNK_SIZE = 512 # Target chunk size in tokens
BATCH_SIZE_EMBEDDING = 64 # Batch size for encoding with sentence-transformer

# REMOVED Gemini API Constants and check

# MODEL_ID = "gemini-2.0-flash"
# BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
#
# if API_KEY is None:
#     raise EnvironmentError("Environment variable 'GEMINI_API_KEY' not set. Export your Gemini API key before running.")

# REMOVED build_batch_judge_prompt function
# def build_batch_judge_prompt(batch_data: List[Dict]) -> str:
#     """
#     Creates a prompt for judging multiple items in one request
#     """
#     # ... (removed function body) ...

# REMOVED process_batch function
# def process_batch(batch_data: List[Dict], client: OpenAI) -> List[Dict]:
#     """
#     Process a batch of items in a single API call
#     """
#     # ... (removed function body) ...

##############################################
# Helper Functions for Embeddings
##############################################

def chunk_text(text: str, tokenizer, chunk_size: int) -> List[str]:
    """Chunks text into segments with a maximum token count."""
    if not text:
        return []
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(tokenizer.convert_tokens_to_string(chunk_tokens))
    # Handle potential empty strings if tokenization results in them
    return [chunk for chunk in chunks if chunk.strip()]


def calculate_similarity(
    question: str,
    text_a: str,
    text_b: str,
    answers: Union[List[str], str],
    model: SentenceTransformer,
    tokenizer,
    chunk_size: int
) -> Dict[str, float]:
    """
    Calculates average cosine similarity between context (question + answers) 
    and chunks of text_a/text_b.
    """
    # 1. Combine question and answers into context
    if isinstance(answers, list):
        answer_text = "\n".join(answers)
    else:
        answer_text = answers if answers else ""
    context_text = f"Question: {question}\n\nAnswers:\n{answer_text}"

    # 2. Get embedding for the combined context
    device = model.device
    context_embedding = model.encode(context_text, convert_to_tensor=True, show_progress_bar=False, device=device)
    # Ensure context_embedding is 2D for util.cos_sim
    if len(context_embedding.shape) == 1:
        context_embedding = context_embedding.unsqueeze(0)

    # 3. Chunk texts A and B
    chunks_a = chunk_text(text_a, tokenizer, chunk_size)
    chunks_b = chunk_text(text_b, tokenizer, chunk_size)

    # 4. Calculate average similarity for Text A
    avg_sim_a = 0.0
    if chunks_a:
        # Embed chunks
        embeddings_a = model.encode(chunks_a, convert_to_tensor=True, show_progress_bar=False, batch_size=BATCH_SIZE_EMBEDDING, device=device)
        # Calculate cosine similarities: shape (1, num_chunks_a)
        cosine_scores_a = util.cos_sim(context_embedding, embeddings_a)
        # Calculate the average score
        if cosine_scores_a.numel() > 0:
            avg_sim_a = torch.mean(cosine_scores_a).item()

    # 5. Calculate average similarity for Text B
    avg_sim_b = 0.0
    if chunks_b:
        # Embed chunks
        embeddings_b = model.encode(chunks_b, convert_to_tensor=True, show_progress_bar=False, batch_size=BATCH_SIZE_EMBEDDING, device=device)
        # Calculate cosine similarities: shape (1, num_chunks_b)
        cosine_scores_b = util.cos_sim(context_embedding, embeddings_b)
        # Calculate the average score
        if cosine_scores_b.numel() > 0:
            avg_sim_b = torch.mean(cosine_scores_b).item()

    # Return average scores
    return {"avg_sim_A": avg_sim_a, "avg_sim_B": avg_sim_b}


##############################################
# Main Logic: judge_data
##############################################
def judge_data(
    input_file: str = DEFAULT_DATA_PATH,
    output_file: str = DEFAULT_OUTPUT_PATH,
    threshold: float = SIM_THRESHOLD,
    chunk_size: int = CHUNK_SIZE,
    model_name: str = EMBEDDING_MODEL_NAME,
    tokenizer_name: str = TOKENIZER_NAME
):
    """
    Reads JSONL, calculates *average* similarity of text chunks against 
    (question + answer) context using embeddings, and processes items.
    Filters based on threshold applied to average scores.
    """
    # Initialize Model and Tokenizer
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"Loading tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if hasattr(model, 'max_seq_length') and chunk_size > model.max_seq_length:
        print(f"Warning: chunk_size ({chunk_size}) exceeds model max length ({model.max_seq_length}). Adjusting chunk_size.")
        chunk_size = model.max_seq_length
    elif not hasattr(model, 'max_seq_length'):
         print(f"Warning: Could not automatically determine model's max sequence length. Assuming chunk_size {chunk_size} is acceptable.")


    # REMOVED OpenAI client initialization
    # client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Could not find the input file: {input_file}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_to_save = []
    lines_count = 0

    print(f"[judge_data] Reading from {input_file}, threshold={threshold}, chunk_size={chunk_size} ...")

    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            lines_count += 1
            item = json.loads(line)

            # Extract data for the current item
            seed_id = item.get('seed_id', '')
            question = item.get('input', '')
            answers = item.get('answers', [])
            text_a = item.get('text_A', '')
            text_b = item.get('text_B', '')

            # Calculate *average* similarity using the updated function
            similarity_scores = calculate_similarity(
                question, text_a, text_b, answers, model, tokenizer, chunk_size
            )
            avg_sim_a = similarity_scores["avg_sim_A"]
            avg_sim_b = similarity_scores["avg_sim_B"]

            print(f"Item {seed_id}: Avg_Sim_A={avg_sim_a:.4f}, Avg_Sim_B={avg_sim_b:.4f}")

            # Filter results based on threshold applied to *average* scores
            if avg_sim_a < threshold or avg_sim_b < threshold:
                record = {
                    "seed_id": str(seed_id),
                    "sim_A": float(avg_sim_a),
                    "sim_B": float(avg_sim_b),
                    "model_id": model_name,
                    "answer_id": shortuuid.uuid()
                }
                results_to_save.append(record)
                print(f"-> Added item {seed_id} to results (average score below threshold {threshold})")

            # Optional: Add progress update every N lines
            if lines_count % 50 == 0:
                 print(f"Processed {lines_count} lines...")

    print(f"[judge_data] Processed {lines_count} lines from {input_file}.")
    print(f"[judge_data] Found {len(results_to_save)} items where average similarity score < {threshold}.")

    with open(output_file, "w", encoding="utf-8") as fout:
        for r in results_to_save:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[judge_data] Wrote {len(results_to_save)} lines to {output_file}")

# -------------------------------------------------------------
# CLI helper when executed as a script
# -------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()
    judge_data()
    elapsed = time.time() - start_time
    print(f"Run completed in {elapsed:.2f} seconds")