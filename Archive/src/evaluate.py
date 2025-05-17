# calls the student model to generate the answers based on the context_A and context_B
# then save the answers as pairs

#!/usr/bin/env python3

"""
Use vLLM to load a local Llama-based model and generate answers from
context_A and context_B for each seed data item. Saves results as a .jsonl file.

All logic is inside the `enumerate_n_items` function so it can be
imported and called from other scripts.
"""

import os
import json
import shortuuid
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

##########################################
# Global Parameters (edit as needed)
##########################################
# Default student model (can be overridden via CLI)
MODEL_NAME_OR_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # placeholder for student model
DTYPE = "auto"
TRUST_REMOTE_CODE = True
TENSOR_PARALLEL_SIZE = 1
MAX_MODEL_LEN = 8000
GPU_MEMORY_UTIL = 0.5

# vLLM sampling / generation params
TEMPERATURE = 0.5
TOP_P = 0.8
TOP_K = 50
MAX_NEW_TOKENS = 1024
REPETITION_PENALTY = 1.0

# Batch size for chunking seed items
BATCH_SIZE = 8

# Default path for input / output
DEFAULT_SEED_DATA_PATH = "data/data_seed.json"
DEFAULT_OUTPUT_PATH = "data/eval.jsonl"


##########################################
# vLLM Imports / Setup
##########################################
try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    raise ImportError("You must install vLLM to run this script: pip install vllm") from e


def load_vllm_model(model_path: str = MODEL_NAME_OR_PATH):
    """
    Initialize the vLLM model with global parameters.
    Returns the LLM object and the tokenizer.
    """
    try:
        # 首先加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            token="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"
        )
        
        # 获取词汇表大小
        vocab_size = len(tokenizer)
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            token="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"
        )
        
        # 调整模型词汇表大小以匹配分词器
        if base_model.get_input_embeddings().weight.shape[0] != vocab_size:
            print(f"Resizing model embeddings from {base_model.get_input_embeddings().weight.shape[0]} to {vocab_size}")
            base_model.resize_token_embeddings(vocab_size)
            
        # 加载模型
        llm = LLM(
            model=model_path,
            dtype=DTYPE,
            trust_remote_code=TRUST_REMOTE_CODE,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTIL,
        )
    except ValueError as e:
        if "No supported config format found" in str(e):
            # if the model is a peft model, we need to load the base model and merge the lora weights
            from peft import PeftModel, PeftConfig
            from safetensors.torch import load_file
            import torch
            
            # load the peft config
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_path = peft_config.base_model_name_or_path
            
            # load the base model and tokenizer
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                trust_remote_code=TRUST_REMOTE_CODE,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=TRUST_REMOTE_CODE
            )
            
            # load the lora weights using safetensors
            lora_state_dict = load_file(os.path.join(model_path, "adapter_model.safetensors"))
            
            # get the vocabulary size from the lora weights
            for key in lora_state_dict.keys():
                if "embed_tokens.weight" in key:
                    vocab_size = lora_state_dict[key].shape[0]
                    break
            
            # if the vocabulary size is not the same, adjust the vocabulary size
            if base_model.get_input_embeddings().weight.shape[0] != vocab_size:
                print(f"Adjusting vocabulary size from {base_model.get_input_embeddings().weight.shape[0]} to {vocab_size}")
                base_model.resize_token_embeddings(vocab_size)
            
            # load the peft model
            model = PeftModel.from_pretrained(
                base_model,
                model_path,
                device_map="auto"
            )
            
            # merge the lora weights to the base model
            model = model.merge_and_unload()
            
            # save the merged model to a temporary directory
            temp_model_path = os.path.join(os.path.dirname(model_path), "merged_model")
            model.save_pretrained(temp_model_path)
            tokenizer.save_pretrained(temp_model_path)
            
            # load the merged model using vLLM
            llm = LLM(
                model=temp_model_path,
                dtype=DTYPE,
                trust_remote_code=TRUST_REMOTE_CODE,
                tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                max_model_len=MAX_MODEL_LEN,
                gpu_memory_utilization=GPU_MEMORY_UTIL,
            )
        else:
            raise e
            
    tok = llm.get_tokenizer()
    return llm, tok

def generate_answers_for_context(
    llm: LLM,
    tokenizer,
    data_items: List[Dict],
    context_key: str,
) -> List[str]:
    """
    For each item in data_items, calls the model with the specified context_key
    ('context_A' or 'context_B') and returns a list of answers (strings).
    Utilizes the global sampling params for vLLM.
    """

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_NEW_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
    )

    all_answers = [None] * len(data_items)
    current_idx = 0

    while current_idx < len(data_items):
        batch_slice = data_items[current_idx : current_idx + BATCH_SIZE]
        prompts = []

        for dp in batch_slice:
            context_val = dp.get(context_key, "")
            question_val = dp.get("input", "")
            lang = dp.get("language", "en")  # default English if not provided
            # Build a concise-answer prompt. Instruct the model to answer strictly from context
            # and to keep the response short (ideally a phrase or single sentence). This helps
            # avoid long hallucinated outputs like the ones observed in previous runs.
            prompt_str = (
                "You are a question-answering assistant. Read the context and answer the question "
                "using ONLY the information in the context. If the answer is not explicitly stated, "
                "reply with the single word 'NOT IN CONTEXT'. Respond with a concise phrase, without "
                "restating the question or adding extra commentary. Output your answer **in the same language as the question**, which is {lang}.\n\n"
                f"Context:\n{context_val}\n\n"
                f"Question: {question_val}\nAnswer:"
            )
            prompts.append(prompt_str)

        # vLLM generate
        results = llm.generate(
            sampling_params=sampling_params,
            prompts=prompts,
        )

        for i, res in enumerate(results):
            if not res.outputs:
                out_text = ""
            else:
                out_text = res.outputs[0].text
            all_answers[current_idx + i] = out_text

        #current_idx += BATCH_SIZE
        current_idx += len(batch_slice)

    return all_answers

def enumerate_n_items(
    seed_data_path: str = DEFAULT_SEED_DATA_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    model_path: str = MODEL_NAME_OR_PATH,
):
    """
    Loads seed data from `seed_data_path`, uses vLLM to generate answers for
    context_A and context_B, then saves the results line-by-line to `output_path`.

    The final JSON lines have this structure:
    {
      "seed_id": "...",
      "input": "...",
      "answers": "...",
      "text_A": "...",
      "text_B": "...",
      "model_id": "...",
      "answer_id": "..."
    }
    """

    # 1) Load the model
    llm, tokenizer = load_vllm_model(model_path)
    model_name = model_path  # record which model produced the answers

    # 2) Load seed data
    if not os.path.exists(seed_data_path):
        raise FileNotFoundError(f"Seed data not found: {seed_data_path}")
    with open(seed_data_path, "r", encoding="utf-8") as f:
        raw_seed = json.load(f)

    # Verify each item already has a non-empty _id. We do NOT generate ids here; the
    # assumption is that the upstream seed-building pipeline (data_prep → generate_seed_data)
    # has assigned a stable id once and for all. If any item is missing an id we stop early—
    # this prevents downstream mismatches.

    seed_data = []
    for idx, dp in enumerate(raw_seed):
        if not dp.get("_id"):
            raise ValueError(
                f"Seed item at index {idx} is missing '_id'. Please regenerate the seed file "
                "with ids before running evaluate.py."
            )
        seed_data.append(dp)
    print(f"Loaded {len(seed_data)} items from {seed_data_path}.")

    # 3) Generate answers for context_A
    print("[enumerate_n_items] Generating answers for context_A...")
    answers_A = generate_answers_for_context(llm, tokenizer, seed_data, context_key="context_A")

    # 4) Generate answers for context_B
    print("[enumerate_n_items] Generating answers for context_B...")
    answers_B = generate_answers_for_context(llm, tokenizer, seed_data, context_key="context_B")

    # 5) Save each record as line-based JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[enumerate_n_items] Saving to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, dp in enumerate(seed_data):
            record = {
                "seed_id": dp.get("_id", ""),
                "input": dp.get("input", ""),
                "answers": dp.get("answers", ""), # ground truth 
                "text_A": answers_A[i],
                "text_B": answers_B[i],
                "model_id": model_name,
                "answer_id": shortuuid.uuid()
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[enumerate_n_items] Done. Wrote {len(seed_data)} lines to {output_path}")

# -------------------------------------------------------------------------
# Optional CLI entry point via python evaluate.py enumerate_n_items --help
# -------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        import fire  # type: ignore
        fire.Fire({"enumerate_n_items": enumerate_n_items})
    except ImportError:
        # Fallback simple run with defaults if fire is not installed
        enumerate_n_items()
