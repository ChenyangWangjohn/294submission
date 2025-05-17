# download the data, reconstruct the data from longbench V1 and save both to specific destinations
# adapt the code from LefengL

import os
import json
from datasets import load_dataset
import re
from rank_bm25 import BM25Okapi
from splitter import split_long_sentence, get_word_len, regex
import shortuuid
import os
from transformers import AutoTokenizer

MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_MODEL_LEN = 8000 

def get_tokenizer():
    """Get the tokenizer for Llama 3 with proper authentication"""
    try:
        # Use your HF token for authentication
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True,
            token=
        )
        print(f"Successfully loaded {MODEL_PATH} tokenizer")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise ValueError("Failed to load tokenizer. ")
    
def calculate_prompt_template_tokens(tokenizer, lang="en"):
    """Calculate the number of tokens used by the prompt template defined in evaluate.py."""
    prompt_str = (
                "You are a question-answering assistant. Read the context and answer the question "
                "using ONLY the information in the context. If the answer is not explicitly stated, "
                "reply with the single word 'NOT IN CONTEXT'. Respond with a concise phrase, without "
                "restating the question or adding extra commentary. Output your answer **in the same language as the question**, which is {lang}.\n\n"
                f"Context:"
                f"Question: Answer:"
            ).format(lang=lang, context="", question="")
    
    return len(tokenizer.encode(prompt_str))

def split_and_rearrange_context(template_tokens_length, tokenizer, context, query=None, chunk_size=512):
    """
    Truncate the original context_A base on query relevanceif neccessary, 
    and create context_B by moving important chunk at the middle.
    
    Args:
        tokenizer: The tokenizer to use for counting tokens
        context (str): Original context text
        query (str): Query text for relevance calculation (required)
        chunk_size (int): Chunk size for BM25 processing, default is 512
        
    Returns:
        tuple: (context_A, context_B) - both within model token limits
    """
    # Every item should have a query(input question)
    if not query:
        raise ValueError("Empty query provided to split_and_rearrange_context. Input questions should be handled before calling this function.")
    
    # Reserved tokens for prompt template, question, etc.
    MARGIN_TOKENS = 20  # allow future question growth
    question_tokens = len(tokenizer.encode(query))
    reserved_tokens = template_tokens_length + question_tokens + MARGIN_TOKENS
    max_context_tokens = MAX_MODEL_LEN - reserved_tokens

    # encode to check original contextlength
    full_tokens = tokenizer.encode(context)
    # If context already fits, return it as-is for both context A and B
    if len(full_tokens) <= max_context_tokens:
        # Use BM25 for relevance-based rearrangement
        # 1. Split text into chunks
        chunks = split_long_sentence(context, regex, chunk_size=chunk_size)
        tokenized_chunks = [get_word_list(chunk) for chunk in chunks]
        tokenized_query = get_word_list(query)
            
        # 2. Calculate relevance using BM25
        bm25 = BM25Okapi(tokenized_chunks)
        scores = bm25.get_scores(tokenized_query)
        # 3. Sort by relevance
        chunk_scores = list(zip(chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
        # 4. Rearrange text using interleaved importance ordering
        sorted_chunks = [chunk for chunk, _ in chunk_scores]
        
        # If there are few chunks, return sorted text directly
        if len(sorted_chunks) <= 4:
            context_B = " ".join(sorted_chunks)
            return context, context_B
        
        # new rearrangement logic to create context_B: Important chunks in the middle
        odd_docs = sorted_chunks[0::2] 
        even_docs = sorted_chunks[1::2]
        reordered_docs = even_docs[::-1] + odd_docs
        context_B = " ".join(reordered_docs)  
        return context, context_B
    # For long contexts that need truncation with query-based relevance
    else:
        chunks = split_long_sentence(context, regex, chunk_size=chunk_size)
        tokenized_chunks = [get_word_list(chunk) for chunk in chunks]
        tokenized_query = get_word_list(query)
        
        bm25 = BM25Okapi(tokenized_chunks)
        scores = bm25.get_scores(tokenized_query)
        
        # Create tuples of (chunk, score, original_index)
        chunk_data = [(chunks[i], scores[i], i) for i in range(len(chunks))]
        
        # Sort by relevance (high to low)
        sorted_by_relevance = sorted(chunk_data, key=lambda x: x[1], reverse=True)
        
        # Create context_A by selecting most relevant chunks until token limit
        selected_chunks = []
        selected_indices = set()
        total_tokens = 0
        
        for chunk, _, idx in sorted_by_relevance:
            chunk_tokens = len(tokenizer.encode(chunk))
            if total_tokens + chunk_tokens <= max_context_tokens:
                selected_chunks.append((chunk, idx))
                selected_indices.add(idx)
                total_tokens += chunk_tokens
            else:
                # Skip least relevant chunks that would exceed token limit
                continue
        
        # Sort selected chunks by original index for context_A
        # context_A is the original text without the least relevant chunks(to fit in model max input length)
        context_A_chunks = [c for c, idx in sorted(selected_chunks, key=lambda x: x[1])]
        context_A = " ".join(context_A_chunks)
        
        # context B is reaaranged to move the most relevant chunks in context A to the middle
        sorted_by_relevance_selected = [item for item in sorted_by_relevance if item[2] in selected_indices]
        # Apply the new rearrangement logic
        odd_docs = [item[0] for i, item in enumerate(sorted_by_relevance_selected) if i % 2 == 0]
        even_docs = [item[0] for i, item in enumerate(sorted_by_relevance_selected) if i % 2 == 1]
        context_B = " ".join(even_docs[::-1] + odd_docs)  # Important chunks in the middle
        
        return context_A, context_B

def get_word_list(s1):
    """Split text into word list"""
    # Separate sentences, Chinese by character, English by word, numbers by space
    regEx = re.compile(r'\W')   
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

def test_prompt_input_length(data, tokenizer, max_length=MAX_MODEL_LEN):
    """Make sure evaluaton prompts (template + question + context) fit within token limits"""
    issues = 0
    max_prompt_tokens = 0
    
    print("\nTesting evaluation prompt lengths...")
    for i, item in enumerate(data):
        # Get the prompt components
        context_a = item["context_A"]
        context_b = item["context_B"]
        question = item["input"]
        _id = item["_id"]
        lang = item.get("language", "en")
        if not lang or lang.strip() == "":
            lang = "en"
        
        prompt_a = (
            "You are a question-answering assistant. Read the context and answer the question "
            "using ONLY the information in the context. If the answer is not explicitly stated, "
            "reply with the single word 'NOT IN CONTEXT'. Respond with a concise phrase, without "
            "restating the question or adding extra commentary. Output your answer **in the same language as the question**, which is {lang}.\n\n"
            f"Context:\n{context_a}\n\n"
            f"Question: {question}\nAnswer:"
        ).format(lang=lang)
        
        prompt_b = (
            "You are a question-answering assistant. Read the context and answer the question "
            "using ONLY the information in the context. If the answer is not explicitly stated, "
            "reply with the single word 'NOT IN CONTEXT'. Respond with a concise phrase, without "
            "restating the question or adding extra commentary. Output your answer **in the same language as the question**, which is {lang}.\n\n"
            f"Context:\n{context_b}\n\n"
            f"Question: {question}\nAnswer:"
        ).format(lang=lang)
        
        prompt_a_tokens = len(tokenizer.encode(prompt_a))
        prompt_b_tokens = len(tokenizer.encode(prompt_b))
        
        max_prompt_tokens = max(max_prompt_tokens, prompt_a_tokens, prompt_b_tokens)
        
        if prompt_a_tokens > max_length:
            print(f"WARNING: Item {_id} prompt_A exceeds token limit: {prompt_a_tokens} > {max_length}")
            issues += 1
            
        if prompt_b_tokens > max_length:
            print(f"WARNING: Item {_id} prompt_B exceeds token limit: {prompt_b_tokens} > {max_length}")
            issues += 1
    
    print(f"Prompt length test complete:")
    print(f"- Max prompt tokens: {max_prompt_tokens}/{max_length}")
    print(f"- Issues found: {issues}/{len(data)*2} prompts")
    
    return issues == 0

def main():
    # Initialize tokenizer
    tokenizer = get_tokenizer()

    template_tokens_length = calculate_prompt_template_tokens(tokenizer)
    print(f"Evaluation Prompt Template tokens length: {template_tokens_length}")
    
    # data_types = ["narrativeqa", "qasper", "multifieldqa_en"]
    data_types = [
        "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "dureader",
        "qmsum", "passage_retrieval_en", "trec", "triviaqa", "samsum",
        "passage_count", "multi_news","gov_report", "gov_report_e", "multi_news_e", "passage_count_e"]  
    # This will hold all processed items from all subdatasets

    subset_input = {
    "gov_report": "Please summarize the following government work report.",
    "gov_report_e": "Please summarize the following government work report.",
    "multi_news": "Please summarize the provided news articles.",
    "multi_news_e": "Please summarize the provided news articles.",
    "passage_count": "Based on the provided text, determine the total number of unique or distinct paragraphs it contains.",
    "passage_count_e": "Based on the provided text, determine the total number of unique or distinct paragraphs it contains.",
    "vcsum": "Please summarize the following meeting records."
}
    combined_data = []
    total_items = 0
    
    # Iterate over each subdataset in the list
    for sub_dataset in data_types:
        # Load the test split for the given subdataset
        ds = load_dataset('THUDM/LongBench', sub_dataset, split="test")
        print(f"Processing subdataset '{sub_dataset}' with {len(ds)} items.")
        num=0
        # Process each item in the loaded subdataset
        for item in ds:
            num+=1
            # Get the original context
            original_context = item.get("context", "")

            # handle empty input
            input_q = item.get("input", "")
            if input_q=='':
                if sub_dataset in subset_input:
                    input_q = subset_input[sub_dataset]
                else:
                    raise ValueError(
                f"Error in dataset '{sub_dataset}': Item (_id: {item.get('_id', '')}   ) has an empty 'input' field, "
                f"but this subset is not defined in 'subset_instructions' to handle empty inputs. ")

            # Truncate the context if neccessary
            # And reconstruct the context using our splitting/reordering function
            original_context, reconstructed_context = split_and_rearrange_context(template_tokens_length, tokenizer, original_context, input_q)

            # Create a new item with both original and reconstructed context
            new_item = {
                "input": input_q,
                "context_A": original_context,       # Original context before reconstruction
                "context_B": reconstructed_context,  # Reconstructed context
                "answers": item.get("answers", ""),
                "question_type": sub_dataset,        # Use the current subdataset name as question type
                "language": item.get("language", ""),
                "use_cot": True,
                "_id": shortuuid.uuid(),             # assign stable unique id
            }
            combined_data.append(new_item)
        
        total_items += len(ds)

    # Make sure the context is truncated to meet the model input length limit
    # test_passed = test_prompt_input_length(combined_data, tokenizer)
    # if not test_passed:
    #     print("WARNING: Some prompts exceed token limits.")
    # else:
    #     print("SUCCESS: All prompts fit within token limits.")

    # 3) Save everything to a single JSON file
    output_file = "data/longbench_C.json"
    if not os.path.exists('data'):
        os.makedirs('data')

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)

    print(f"Saved combined dataset with both contexts to '{output_file}'. Total items: {len(combined_data)}")

if __name__ == "__main__":
    main()
