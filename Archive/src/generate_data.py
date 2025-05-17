# Using the LongWriter-llama3.1-8b to generate new data based on the seed data
# The longwriter will be our new teacher model that in charge of generating the new data
"""
Adapted from https://github.com/tatsu-lab/stanford_alpaca
"""
"""
data_generation_vllm.py

Adapted from https://github.com/tatsu-lab/stanford_alpaca and LLM2LLM code,
but replaced the OpenAI calls with vLLM-based local Llama logic.

Usage:
    # CLI
    python data_generation_vllm.py generate_data \\
        --seed_tasks_path="data/data_seed.json" \\
        --output_dir="data" \\
        --instructions_per_seed_task=4 \\
        --similarity_threshold=0.9 \\
        --chunk_size=512 \\
        --chunk_match_ratio_threshold=0.8 \\
        ...

    # Import from another Python script
    from data_generation_vllm import generate_data

    generate_data(
        seed_tasks_path="data/data_seed.json",
        output_dir="data",
        instructions_per_seed_task=4,
        similarity_threshold=0.9,
        chunk_size=512,
        chunk_match_ratio_threshold=0.8,
        ...
    )
"""

import os
import json
import random
import re
import string
import time
import gc
import shortuuid
from functools import partial
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Tuple

# For command-line usage
import fire
import numpy as np
import tqdm
from splitter import split_long_sentence, get_word_len, regex
# Sentence Transformer imports
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import PreTrainedTokenizerBase # For type hinting embed_model.tokenizer

# Re-add BM25Okapi if split_and_rearrange_context uses it
from rank_bm25 import BM25Okapi


# (Optional) adjust import paths if needed
import sys
from pathlib import Path
utils_path = str(Path(__file__).resolve().parent.parent)
if utils_path not in sys.path:
    sys.path.append(utils_path)

import utils  # Possibly your local utilities

############################################################
# vLLM imports
############################################################
try:
    from vllm import LLM, SamplingParams
    # HuggingFace tokenizer for CPU-side work
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("You must install vLLM and transformers to run this code (pip install vllm transformers)")

############################################################
# Global Model Setup (replacing GPT)
############################################################
MODEL_PATH = "THUDM/LongWriter-llama3.1-8b"

# vLLM optimization settings
ENFORCE_EAGER = False      # avoid slow CUDA graph capture
GPU_MEM_UTIL = 0.85
MAX_CTX_LEN = 8192        # max context length, reduce if running out of memory

llm = LLM(
    model=MODEL_PATH,
    dtype="bfloat16",
    trust_remote_code=True,
    max_model_len=MAX_CTX_LEN,
    gpu_memory_utilization=GPU_MEM_UTIL,
    enforce_eager=ENFORCE_EAGER,
    tokenizer_pool_size=4,  # parallel tokenization
    enable_chunked_prefill=True,
)
# Use a separate HF tokenizer (vLLM tokenizer pool not compatible with direct access)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)

# Sentence Transformer model for similarity checks
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
# Get the tokenizer from the Sentence Transformer model for chunking
embed_tokenizer: PreTrainedTokenizerBase = embed_model.tokenizer


# Base sampling params (length-agnostic). We will override max/min_tokens per call.
BASE_SAMPLING_PARAMS = dict(
    temperature=0.5,
    top_p=0.8,
    top_k=50,
    repetition_penalty=1.0,
)

############################################################
# Text Chunking Helper
############################################################
def chunk_text_by_tokens(text: str, tokenizer: PreTrainedTokenizerBase, chunk_size: int) -> List[str]:
    """Chunks text into segments with a maximum token count."""
    if not text:
        return []
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        # Decode chunk tokens back to string. skip_special_tokens=True is important.
        # clean_up_tokenization_spaces=False might be needed depending on tokenizer behavior
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if chunk_text.strip(): # Avoid adding empty chunks
             chunks.append(chunk_text)
        start_idx = end_idx
        
    # Handle case where text is shorter than chunk_size or results in one chunk
    if not chunks and text.strip():
         return [text]
         
    return chunks

############################################################
# Chunked Similarity Comparison Helper
############################################################
def compare_chunked_embeddings(
    chunks_a_embeds: Optional[torch.Tensor],
    chunks_b_embeds: Optional[torch.Tensor],
    similarity_threshold: float,
    chunk_match_ratio_threshold: float
) -> bool:
    """
    Compares two sets of chunk embeddings.
    Returns True if they are considered similar based on chunk matches.
    """
    # Handle cases where one or both texts couldn't be embedded (e.g., empty text)
    if chunks_a_embeds is None or chunks_b_embeds is None:
        return False # Cannot compare if one is missing
    if chunks_a_embeds.shape[0] == 0 or chunks_b_embeds.shape[0] == 0:
        return False # Cannot compare if one has no chunks

    # Compare embeddings of corresponding chunks
    len_a = chunks_a_embeds.shape[0]
    len_b = chunks_b_embeds.shape[0]
    num_comparisons = min(len_a, len_b)
    
    if num_comparisons == 0:
         return False # Should not happen if we checked shape[0] > 0 above, but safety first.

    matching_chunks = 0
    for i in range(num_comparisons):
        # Calculate cosine similarity between the i-th chunk of A and B
        similarity = util.cos_sim(chunks_a_embeds[i], chunks_b_embeds[i]).item()
        if similarity >= similarity_threshold:
            matching_chunks += 1

    # Check if the ratio of matching chunks meets the threshold
    match_ratio = matching_chunks / num_comparisons
    return match_ratio >= chunk_match_ratio_threshold


############################################################
# Reconstruct function, prompt template, etc.
############################################################

def split_and_rearrange_context(context, query=None, chunk_size=512):
    """
    Rearrange context based on query relevance.
    If a query is provided, rearrange by relevance; otherwise use the default middle rearrangement strategy.
    
    Args:
        context (str): Original context text
        query (str, optional): Query text for relevance calculation
        chunk_size (int): Chunk size, default is 512
        
    Returns:
        str: Rearranged text
    """
    # If no query is provided, use the default middle rearrangement strategy
    if not query:
        words = context.split()
        if len(words) < 4:
            return context
        
        quarter = len(words) // 4
        beginning = words[:quarter]
        first_half_middle = words[quarter : 2 * quarter]
        second_half_middle = words[2 * quarter : 3 * quarter]
        end = words[3 * quarter:]
        
        rearranged = first_half_middle + beginning + end + second_half_middle
        return " ".join(rearranged)
    
    # Use BM25 for relevance-based rearrangement
    # 1. Split text into chunks (using sentence splitter, not token chunker)
    # NOTE: Using the 'splitter.py' split_long_sentence here, not the token chunker
    sentence_chunks = split_long_sentence(context, regex, chunk_size=chunk_size)

    if not sentence_chunks:
         return context # Cannot rearrange if no chunks

    # 2. Calculate relevance using BM25
    # Make sure BM25Okapi is imported
    try:
        tokenized_chunks = [get_word_list(chunk) for chunk in sentence_chunks]
        tokenized_query = get_word_list(query)

        # Handle empty lists to avoid BM25 errors
        if not tokenized_chunks or not tokenized_query:
             print(f"Warning: Could not tokenize chunks or query for BM25 rearrangement. Context: {context[:100]}..., Query: {query}")
             return context # Fallback to original context

        bm25 = BM25Okapi(tokenized_chunks)
        scores = bm25.get_scores(tokenized_query)

        # 3. Sort by relevance
        chunk_scores = list(zip(sentence_chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        # 4. Rearrange text using interleaved importance ordering
        sorted_chunks = [chunk for chunk, _ in chunk_scores]

        # If there are few chunks, return sorted text directly
        if len(sorted_chunks) <= 4:
            return " ".join(sorted_chunks)

        odd_docs = sorted_chunks[0::2]
        even_docs = sorted_chunks[1::2]
        reordered_docs = odd_docs + even_docs[::-1]

        return " ".join(reordered_docs)
    except Exception as e:
        print(f"Error during BM25 rearrangement: {e}. Falling back to original context.")
        print(f"Context: {context[:100]}..., Query: {query}")
        return context # Fallback


def get_word_list(s1):
    """Split text into word list"""
    # Separate sentences, Chinese by character, English by word, numbers by space
    regEx = re.compile('[\W]')   
    res = re.compile(r"([\u4e00-\u9fa5])")    # [\u4e00-\u9fa5] for Chinese
    
    p1 = regEx.split(s1.lower())
    str1_list = []
    for str in p1:
        if res.split(str) == None:
            str1_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str1_list.append(ch)
    
    return [w for w in str1_list if len(w.strip()) > 0]

def create_prompt_for_new_contextA(seed_item, attempt=0):
    """
    A simple prompt for creating a new 'context_A' from the seed.
    """
    user_text = f"""You have a question with input: {seed_item.get('input', '')}
Answers: {seed_item.get('answers', '')}
Language: {seed_item.get('language', '')}
Question Type: {seed_item.get('question_type', '')}

Please generate a new 'context_A' that is thematically similar but not identical.
Attempt: {attempt}
"""
    return f"[INST]{user_text}[/INST]"


# Keep this helper function to structure the final output dict
def create_new_item_dict(seed_item, new_contextA):
    """
    Helper to structure the final item dict after context_A is generated and filtered.
    Calls split_and_rearrange_context to generate context_B.
    """
    new_contextB = split_and_rearrange_context(new_contextA, query=seed_item.get("input", ""))
    return {
        "input": seed_item.get("input", ""),
        "context_A": new_contextA,
        "context_B": new_contextB,
        "answers": seed_item.get("answers", ""),
        "question_type": seed_item.get("question_type", ""),
        "language": seed_item.get("language", ""),
        "use_cot": seed_item.get("use_cot", True),
        "_id": shortuuid.uuid(), # Generate new ID for the augmented item
        "original_seed_id": seed_item.get("_id") # Keep track of the seed it came from
    }


############################################################
# The main function you can call from other files
############################################################
def generate_data(
    seed_tasks_path="data/data_seed.json",
    output_dir="data",
    instructions_per_seed_task=5,   # generate 5 samples per seed
    similarity_threshold=0.9,       # Use cosine similarity threshold
    chunk_size: int = 512,                 # Max tokens per chunk for similarity
    chunk_match_ratio_threshold: float = 0.8, # Min ratio of matching chunks
    batch_size=16,          # how many prompts to send to vLLM in one go
    flush_every=100,        # How often to save partial results
    max_new_items=None,     # None means no cap
):
    """
    1) Load a seed dataset from `seed_tasks_path`.
    2) For each item, generate 'instructions_per_seed_task' new context_A candidates.
    3) Filter out near-duplicates using Sentence Transformer cosine similarity against existing contexts.
    4) Construct the full item (including rearranged context_B).
    5) Save partial results to 'regen.json' in `output_dir`.
    """
    global embed_model, embed_tokenizer # Make sure we use the global models

    # 1) Load seed tasks
    if not os.path.isfile(seed_tasks_path):
        raise FileNotFoundError(f"Seed task file '{seed_tasks_path}' does not exist. Please provide a valid path.")

    with open(seed_tasks_path, "r", encoding="utf-8") as f:
        seed_tasks = json.load(f)
    print(f"Loaded {len(seed_tasks)} seed items from {seed_tasks_path}")

    # 2) Setup output
    os.makedirs(output_dir, exist_ok=True)
    regen_path = os.path.join(output_dir, "regen.json")
    if os.path.exists(regen_path):
        try:
            with open(regen_path, "r", encoding="utf-8") as rf:
                machine_instruction_data = json.load(rf)
            print(f"Loaded {len(machine_instruction_data)} previously generated items from {regen_path}")
        except json.JSONDecodeError:
             print(f"Warning: {regen_path} is corrupted or empty. Starting with an empty list.")
             machine_instruction_data = []
    else:
        machine_instruction_data = []


    # Collect all existing context_A texts to build the initial embedding corpus
    existing_contextA_texts = [d.get("context_A", "") for d in seed_tasks if d.get("context_A")]
    existing_contextA_texts.extend([d.get("context_A", "") for d in machine_instruction_data if d.get("context_A")])

    print(f"Calculating embeddings for {len(existing_contextA_texts)} existing contexts...")
    # Compute initial embeddings in batches for efficiency
    corpus_chunk_embeddings: List[Optional[torch.Tensor]] = [] # Store list of chunk tensors per text
    print(f"Chunking and calculating embeddings for {len(existing_contextA_texts)} existing contexts (chunk size: {chunk_size})...")
    
    # Batch texts for chunking and embedding
    all_existing_chunks = []
    text_indices_for_chunks = [] # Track which text each chunk belongs to
    texts_processed = 0
    for i, text in enumerate(tqdm.tqdm(existing_contextA_texts, desc="Chunking existing texts")):
        chunks = chunk_text_by_tokens(text, embed_tokenizer, chunk_size)
        if chunks:
            all_existing_chunks.extend(chunks)
            text_indices_for_chunks.extend([i] * len(chunks)) # Store original text index for each chunk
            texts_processed += 1
        else:
             # Placeholder if text is empty or cannot be chunked
             corpus_chunk_embeddings.append(None) 
             
    print(f"Calculating embeddings for {len(all_existing_chunks)} chunks from {texts_processed} texts...")
    if all_existing_chunks:
         chunk_embeddings_flat = embed_model.encode(all_existing_chunks, convert_to_tensor=True, show_progress_bar=True)
         
         # Reconstruct the list of tensors, one tensor per original text
         current_text_idx = -1
         temp_chunk_list = []
         corpus_chunk_embeddings_temp = [] # Build the final list here
         for chunk_idx, original_text_idx in enumerate(text_indices_for_chunks):
              if original_text_idx != current_text_idx:
                   # Starting a new text
                   if temp_chunk_list: # Save the completed list for the previous text
                        corpus_chunk_embeddings_temp.append(torch.stack(temp_chunk_list))
                   # Handle texts that had no valid chunks before this one
                   while len(corpus_chunk_embeddings_temp) < original_text_idx:
                       corpus_chunk_embeddings_temp.append(None)
                   temp_chunk_list = [] # Start new list
                   current_text_idx = original_text_idx
              temp_chunk_list.append(chunk_embeddings_flat[chunk_idx])
         
         # Add the last list
         if temp_chunk_list:
             corpus_chunk_embeddings_temp.append(torch.stack(temp_chunk_list))
         # Handle trailing texts that had no valid chunks
         while len(corpus_chunk_embeddings_temp) < len(existing_contextA_texts):
             corpus_chunk_embeddings_temp.append(None)
             
         corpus_chunk_embeddings = corpus_chunk_embeddings_temp # Assign the fully constructed list
         del chunk_embeddings_flat, corpus_chunk_embeddings_temp, temp_chunk_list # Cleanup
         gc.collect()
         torch.cuda.empty_cache()

    print(f"Initial corpus size: {len(corpus_chunk_embeddings)} entries (some might be None).")

    #################### Batched generation with periodic flush #####################
    start_len = len(machine_instruction_data)
    new_items_generated_total = 0
    generated_this_flush_cycle = 0

    all_prompts = []
    all_seed_refs = [] # Keep track of which seed corresponds to each prompt

    print("Preparing generation prompts...")
    for s in tqdm.tqdm(seed_tasks, desc="Preparing Prompts"):
        for attempt in range(instructions_per_seed_task):
            all_prompts.append(create_prompt_for_new_contextA(s, attempt=attempt))
            all_seed_refs.append(s)
            # Early exit if max_new_items cap is likely reached by prompts prepared
            if max_new_items is not None and len(all_prompts) >= max_new_items * 2 : # *2 is heuristic buffer
                 break
        if max_new_items is not None and len(all_prompts) >= max_new_items * 2:
            print(f"Reached prompt limit related to max_new_items ({max_new_items}). Stopping prompt preparation.")
            break


    print(f"Generated {len(all_prompts)} prompts in total.")

    #######################################################################
    # 1) Bulk Generation – Process prompts in batches                     #
    #######################################################################
    generated_outputs: List[tuple[dict, str]] = [] # Store (seed_ref, generated_text)

    print("Starting vLLM generation...")
    for i in tqdm.tqdm(range(0, len(all_prompts), batch_size), desc="vLLM Generation Batches"):
        batch_prompts = all_prompts[i : i + batch_size]
        batch_seed_refs = all_seed_refs[i : i + batch_size]

        if not batch_prompts:
            continue

        # Estimate max/min tokens based on longest prompt in batch? Or fixed reasonable value?
        # Using fixed values for simplicity now.
        max_tok_batch = 6000 # Adjusted from 8192 to be safer for varied prompt lengths
        min_tok = max(32, int(max_tok_batch * 0.5)) # Require at least half

        sampling_params = SamplingParams(
            **BASE_SAMPLING_PARAMS,
            max_tokens=max_tok_batch,
            min_tokens=min_tok,
            # skip_special_tokens=True, # Generally good idea
            # stop=["[INST]", "[/INST]"] # Stop generation if it hallucinates instruction tokens
        )

        try:
            results = llm.generate(prompts=batch_prompts, sampling_params=sampling_params)

            for idx_in_batch, res in enumerate(results):
                if res.outputs:
                    seed_obj = batch_seed_refs[idx_in_batch]
                    out_text = res.outputs[0].text.strip()
                    if out_text: # Basic check for non-empty generation
                        generated_outputs.append((seed_obj, out_text))

        except Exception as e:
            print(f"\nError during vLLM generation batch starting at index {i}: {e}")
            print(f"Problematic prompts (first 5): {batch_prompts[:5]}")
            # Optionally: add more robust error handling, e.g., retries or skipping problematic batches

        # Optional: Early exit from generation loop if max_new_items is hit
        if max_new_items is not None and (start_len + new_items_generated_total + generated_this_flush_cycle >= max_new_items):
             print(f"Reached max_new_items ({max_new_items}) during generation. Stopping early.")
             break # Exit generation loop

    print(f"vLLM generation complete. Collected {len(generated_outputs)} raw outputs.")
    del results # Free GPU memory
    gc.collect()
    torch.cuda.empty_cache()


    #######################################################################
    # 2) Post-processing – Filter duplicates using embeddings             #
    #######################################################################
    print("Starting post-processing and similarity filtering...")
    newly_accepted_items: List[Dict[str, Any]] = []
    candidate_texts = [text for _, text in generated_outputs]
    candidate_seeds = [seed for seed, _ in generated_outputs]

    if not candidate_texts:
        print("No valid outputs generated. Exiting.")
        return

    print(f"Calculating embeddings for {len(candidate_texts)} generated candidates...")
    candidate_chunk_embeddings: List[Optional[torch.Tensor]] = []
    print(f"Chunking and calculating embeddings for {len(candidate_texts)} generated candidates (chunk size: {chunk_size})...")
    
    all_candidate_chunks = []
    cand_text_indices_for_chunks = []
    cands_processed = 0
    for i, text in enumerate(tqdm.tqdm(candidate_texts, desc="Chunking candidates")):
        chunks = chunk_text_by_tokens(text, embed_tokenizer, chunk_size)
        if chunks:
            all_candidate_chunks.extend(chunks)
            cand_text_indices_for_chunks.extend([i] * len(chunks))
            cands_processed += 1
        else:
             candidate_chunk_embeddings.append(None)

    print(f"Calculating embeddings for {len(all_candidate_chunks)} chunks from {cands_processed} candidates...")
    if all_candidate_chunks:
        cand_chunk_embeddings_flat = embed_model.encode(all_candidate_chunks, convert_to_tensor=True, show_progress_bar=True)
        
        # Reconstruct list of tensors
        current_cand_idx = -1
        temp_cand_chunk_list = []
        candidate_chunk_embeddings_temp = []
        for chunk_idx, original_cand_idx in enumerate(cand_text_indices_for_chunks):
             if original_cand_idx != current_cand_idx:
                 if temp_cand_chunk_list:
                     candidate_chunk_embeddings_temp.append(torch.stack(temp_cand_chunk_list))
                 while len(candidate_chunk_embeddings_temp) < original_cand_idx:
                     candidate_chunk_embeddings_temp.append(None)
                 temp_cand_chunk_list = []
                 current_cand_idx = original_cand_idx
             temp_cand_chunk_list.append(cand_chunk_embeddings_flat[chunk_idx])
             
        if temp_cand_chunk_list:
             candidate_chunk_embeddings_temp.append(torch.stack(temp_cand_chunk_list))
        while len(candidate_chunk_embeddings_temp) < len(candidate_texts):
             candidate_chunk_embeddings_temp.append(None)
             
        candidate_chunk_embeddings = candidate_chunk_embeddings_temp
        del cand_chunk_embeddings_flat, candidate_chunk_embeddings_temp, temp_cand_chunk_list
        gc.collect()
        torch.cuda.empty_cache()

    print("Filtering based on similarity...")
    accepted_indices = []
    # Compute cosine similarities between all candidates and the existing corpus
    # Use util.semantic_search for potentially faster search if corpus is large,
    # but pairwise cosine_scores is simpler for moderate sizes.
    # --- Removed incorrect cosine similarity calculation on the whole list ---
    # if corpus_chunk_embeddings.shape[0] > 0: # ERROR: list has no shape
    #     cosine_scores = util.cos_sim(candidate_chunk_embeddings, corpus_chunk_embeddings)
    # else:
    #     # If corpus is empty (first run), accept all non-empty candidates initially
    #     cosine_scores = torch.zeros((candidate_chunk_embeddings.shape[0], 0), device=candidate_chunk_embeddings.device)


    temp_accepted_chunk_embeddings: List[Optional[torch.Tensor]] = [] # Store chunk embeddings of accepted items in this batch

    for i in tqdm.tqdm(range(len(candidate_texts)), desc="Filtering Candidates"):
        candidate_chunks_i = candidate_chunk_embeddings[i]
        
        if candidate_chunks_i is None: # Skip if candidate couldn't be chunked/embedded
             continue

        # Check against existing corpus (seed + previous regen)
        is_too_similar_to_existing = False
        # Iterate through each item in the existing corpus embeddings list
        for corpus_chunks_j in corpus_chunk_embeddings:
             # Compare candidate i with existing item j using the chunked comparison function
             if compare_chunked_embeddings(candidate_chunks_i, corpus_chunks_j, similarity_threshold, chunk_match_ratio_threshold):
                 is_too_similar_to_existing = True
                 break # Found a similar item in existing corpus, no need to check further
        if is_too_similar_to_existing:
            continue # Move to the next candidate

        # Check against candidates already accepted *in this batch*
        is_too_similar_to_batch = False
        # Iterate through embeddings accepted in this batch so far
        for temp_chunks_k in temp_accepted_chunk_embeddings:
             # Compare candidate i with previously accepted candidate k in this batch
             if compare_chunked_embeddings(candidate_chunks_i, temp_chunks_k, similarity_threshold, chunk_match_ratio_threshold):
                 is_too_similar_to_batch = True
                 break # Found similar item in this batch
        if is_too_similar_to_batch:
            continue # Move to the next candidate

        # If it passed both checks, accept it
        accepted_indices.append(i)
        
        # Construct the final item dict and add to accepted list
        seed_ref = candidate_seeds[i]
        accepted_text = candidate_texts[i]
        new_item = create_new_item_dict(seed_ref, accepted_text)
        newly_accepted_items.append(new_item)
        
        # Add its chunk embeddings to the temporary list for subsequent checks in this batch
        temp_accepted_chunk_embeddings.append(candidate_chunks_i)

        # Check if max_new_items cap is reached
        # Use newly_accepted_items list length for accurate count in this cycle
        current_total_items = start_len + new_items_generated_total + len(newly_accepted_items)
        if max_new_items is not None and current_total_items >= max_new_items:
            print(f"Reached max_new_items ({max_new_items}) during filtering. Stopping.")
            break # Exit filtering loop

    print(f"Accepted {len(newly_accepted_items)} new items after filtering.")

    # Add the newly accepted items to the main list
    machine_instruction_data.extend(newly_accepted_items)
    new_items_generated_total += len(newly_accepted_items)

    # Update the main corpus chunk embeddings list with the newly accepted ones
    # Note: We are extending the list, not concatenating tensors here
    corpus_chunk_embeddings.extend(temp_accepted_chunk_embeddings)
    print(f"Updated corpus size: {len(corpus_chunk_embeddings)} entries.")


    # Save the final results
    with open(regen_path, "w", encoding="utf-8") as wf:
        json.dump(machine_instruction_data, wf, ensure_ascii=False, indent=2)
    print(f"[generate_data] Saved {len(machine_instruction_data)} total items to {regen_path}")

    print(f"Generation complete. Added {new_items_generated_total} new items. Total in regen: {len(machine_instruction_data)}")

    # Clean up GPU memory explicitly
    del candidate_chunk_embeddings, temp_accepted_chunk_embeddings
    # corpus_chunk_embeddings is needed if the function is called again in same process, maybe don't delete?
    # Let's keep corpus_chunk_embeddings for now
    # if 'cosine_scores' in locals(): del cosine_scores # Already removed
    gc.collect()
    torch.cuda.empty_cache()


############################################################
# Optional: command-line usage with fire
############################################################
def main(task="generate_data", **kwargs):
    """
    Command-line entry point.
    Example: python generate_data.py generate_data --seed_tasks_path="seeds.json" --similarity_threshold=0.85 --chunk_match_ratio_threshold=0.75
    """
    # Map CLI arguments to function parameters, handling potential name changes
    if 'rouge_score_threshold' in kwargs and 'similarity_threshold' not in kwargs:
         kwargs['similarity_threshold'] = kwargs.pop('rouge_score_threshold')
         print("Warning: 'rouge_score_threshold' used, mapping to 'similarity_threshold'. Please update CLI arguments.")
    if 'generation_workers' in kwargs:
        kwargs.pop('generation_workers') # Remove unused argument
        print("Warning: 'generation_workers' argument is deprecated and ignored.")


    if task == "generate_data":
        # Filter kwargs to only those accepted by generate_data
        accepted_args = ["seed_tasks_path", "output_dir", "instructions_per_seed_task",
                         "similarity_threshold", "chunk_size", "chunk_match_ratio_threshold",
                         "batch_size", "flush_every", "max_new_items"]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_args}
        # Ensure types for CLI args
        if 'chunk_size' in filtered_kwargs: filtered_kwargs['chunk_size'] = int(filtered_kwargs['chunk_size'])
        if 'similarity_threshold' in filtered_kwargs: filtered_kwargs['similarity_threshold'] = float(filtered_kwargs['similarity_threshold'])
        if 'chunk_match_ratio_threshold' in filtered_kwargs: filtered_kwargs['chunk_match_ratio_threshold'] = float(filtered_kwargs['chunk_match_ratio_threshold'])
        if 'instructions_per_seed_task' in filtered_kwargs: filtered_kwargs['instructions_per_seed_task'] = int(filtered_kwargs['instructions_per_seed_task'])
        if 'batch_size' in filtered_kwargs: filtered_kwargs['batch_size'] = int(filtered_kwargs['batch_size'])
        if 'flush_every' in filtered_kwargs: filtered_kwargs['flush_every'] = int(filtered_kwargs['flush_every'])
        if 'max_new_items' in filtered_kwargs and filtered_kwargs['max_new_items'] is not None: filtered_kwargs['max_new_items'] = int(filtered_kwargs['max_new_items'])


        return generate_data(**filtered_kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")

if __name__ == "__main__":
    fire.Fire(main)
