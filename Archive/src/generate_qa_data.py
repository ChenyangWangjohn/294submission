import os, json, random, shortuuid, gc, re, torch, tqdm, fire
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, PreTrainedTokenizerBase, pipeline
from vllm import LLM, SamplingParams
from rank_bm25 import BM25Okapi
from collections import defaultdict
import numpy as np

# ---------------------------------------------------------------------
# 1.  Models (reuse the global instances so CUDA memory is shared)
# ---------------------------------------------------------------------
MODEL_PATH = "THUDM/LongWriter-llama3.1-8b"
_llm_singleton: Optional[LLM] = None
_tokenizer_singleton: Optional[PreTrainedTokenizerBase] = None
_embed_singleton: Optional[SentenceTransformer] = None
def get_llm():
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLM(
            model=MODEL_PATH, dtype="bfloat16",
            max_model_len=8192, gpu_memory_utilization=0.85,
            trust_remote_code=True, enable_chunked_prefill=True,
        )
    return _llm_singleton
def get_tokenizer():
    global _tokenizer_singleton
    if _tokenizer_singleton is None:
        _tokenizer_singleton = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, use_fast=True
        )
    return _tokenizer_singleton
def get_embedder():
    global _embed_singleton
    if _embed_singleton is None:
        _embed_singleton = SentenceTransformer("BAAI/bge-m3")
    return _embed_singleton

# ---------------------------------------------------------------------
# 2.  Prompt construction
# ---------------------------------------------------------------------
QA_GEN_TEMPLATE = """\
You are given a passage delimited by ‹context› tags.
Return a JSON list with up to {n} objects, each object MUST have:
  "q": a question answerable *only* from coloured text,
  "a": the concise answer (<= 100 tokens) copied verbatim.

Rules:
• No two questions may be near-duplicates.
• Answer string must appear intact in the passage.
• Output *only* valid JSON.

‹context›
{ctx}
‹/context›
"""

# Few-shot JSON block to steer the LLM to correct format (no long context)
FEW_SHOT_JSON = """Example:\n[
  {\"q\": \"In which year did the project launch?\", \"a\": \"2015\"},\n  {\"q\": \"Who chaired the committee?\", \"a\": \"Professor Li Ming\"}\n]\n###\n"""

###############################################
# Helper – locate anchor blocks around answer
###############################################

def _sent_tokenize(text: str) -> List[str]:
    """Very lightweight sentence splitter – period, question, exclamation, Chinese 。？！"""
    sents = re.split(r"(?<=[。！？.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def _find_all(sub: str, text: str) -> List[int]:
    """Return list of start indices where `sub` occurs in `text` (case-insensitive)."""
    sub_l, text_l = sub.lower(), text.lower()
    starts = []
    idx = text_l.find(sub_l)
    while idx != -1:
        starts.append(idx)
        idx = text_l.find(sub_l, idx + 1)
    return starts


def _rank_hits_by_similarity(
    hits: List[tuple[int, int]],  # (start_char, end_char)
    context: str,
    query: str,
    embedder: SentenceTransformer,
) -> List[tuple[int, int, float]]:
    """Return list of (start,end,score) sorted by score desc."""
    # Build corpus of snippets (+/- one sentence)
    corpus = []
    for s, e in hits:
        # Take +- one sentence context around the hit for similarity calc
        # Find sentence boundaries
        before = context[:s]
        after = context[e:]
        before_sents = _sent_tokenize(before)
        after_sents = _sent_tokenize(after)
        snippet = " ".join(before_sents[-1:] + [context[s:e]] + after_sents[:1])
        corpus.append(snippet)

    if not corpus:
        return []

    corpus_emb = embedder.encode(corpus, convert_to_tensor=True)
    query_emb = embedder.encode([query], convert_to_tensor=True)
    dense_scores = util.cos_sim(query_emb, corpus_emb)[0].cpu().tolist()  # list of floats

    # BM25 score (simple whitespace tokenization)
    tokenized_corpus = [snip.split() for snip in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    query_toks = query.split()
    bm25_scores = bm25.get_scores(query_toks)

    ranked = []
    for (s, e), ds, bs in zip(hits, dense_scores, bm25_scores):
        ranked.append((s, e, (ds + bs) / 2.0))

    # sort desc by score
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


def _select_anchors(ranked_hits: List[tuple[int, int, float]]) -> List[tuple[int, int]]:
    """Apply heuristics described in design to prune hits."""
    if not ranked_hits:
        return []
    s_max = ranked_hits[0][2]
    # keep hits whose score >= 1/3 * S_max
    filtered = [h for h in ranked_hits if h[2] >= s_max / 3.0]
    # if only first hit has score >= 0.9 * S_max -> use single anchor
    strong_hits = [h for h in filtered if h[2] >= 0.9 * s_max]
    if len(strong_hits) == 1:
        return [(strong_hits[0][0], strong_hits[0][1])]
    return [(h[0], h[1]) for h in filtered]


def _char_to_token_ranges(offset_mapping: List[tuple[int, int]], spans: List[tuple[int, int]]) -> List[tuple[int, int]]:
    """Map character spans to token index spans using tokenizer offset mapping"""
    token_ranges = []
    for s_char, e_char in spans:
        # find first token whose start <= s_char < end
        start_tok = None
        end_tok = None
        for idx, (tok_s, tok_e) in enumerate(offset_mapping):
            if start_tok is None and tok_s <= s_char < tok_e:
                start_tok = idx
            if tok_s < e_char <= tok_e:
                end_tok = idx + 1  # exclusive
                break
        if start_tok is not None and end_tok is not None:
            token_ranges.append((start_tok, end_tok))
    return token_ranges


def extract_anchor_blocks(
    context: str,
    answers: Union[List[str], str],
    query: str,
    tokenizer: PreTrainedTokenizerBase,
    embedder: SentenceTransformer,
    base_block: int = 256,
    merge_block_tokens: int = 512,
) -> List[str]:
    """Return list of anchor block texts (already detokenized)."""
    if isinstance(answers, str):
        answers = [answers]
    # Step 1: find answer hits
    hits = []
    for ans in answers:
        for start in _find_all(ans, context):
            hits.append((start, start + len(ans)))
    if not hits:
        return []  # cannot find answers
    # Step 1b rank
    ranked = _rank_hits_by_similarity(hits, context, query, embedder)
    # Step 1c heuristic selection
    selected_spans = _select_anchors(ranked)
    if not selected_spans:
        return []
    # Step 1d tokenization mapping
    enc = tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)
    offset_map = enc["offset_mapping"]
    token_spans = _char_to_token_ranges(offset_map, selected_spans)
    if not token_spans:
        return []
    # snap to multiples of base_block
    snapped = []
    for st, ed in token_spans:
        new_start = (st // base_block) * base_block
        new_end = ((ed + base_block - 1) // base_block) * base_block
        snapped.append((new_start, new_end))
    # merge overlapping or adjacent into ≤ merge_block_tokens
    snapped.sort()
    merged = []
    for st, ed in snapped:
        if not merged:
            merged.append([st, ed])
        else:
            prev_st, prev_ed = merged[-1]
            if st <= prev_ed and (ed - prev_st) <= merge_block_tokens:
                merged[-1][1] = max(prev_ed, ed)
            else:
                merged.append([st, ed])
    # decode
    anchor_blocks = [tokenizer.decode(enc['input_ids'][st:ed], skip_special_tokens=True, clean_up_tokenization_spaces=True) for st, ed in merged]
    return anchor_blocks


###############################################
# Helper – build masked context with tags
###############################################

def build_masked_context(anchor_blocks: List[str]) -> str:
    out_lines = []
    for idx, block in enumerate(anchor_blocks, start=1):
        out_lines.append(f"<ANCHOR_{idx}> {block} </ANCHOR_{idx}>")
        if idx != len(anchor_blocks):
            out_lines.append(f"### CONTENT OMITTED ###")
    return "\n".join(out_lines)

###############################################
# Updated Prompt builder using anchor blocks
###############################################

def build_prompt(seed_item: Dict[str, Any], n: int, attempt: int, tokenizer, embedder) -> tuple[str, List[str]]:
    """Return prompt string and anchor blocks used (for later validation)."""
    context = seed_item["context_A"]
    answers = seed_item.get("answers", "")
    query = seed_item.get("input", "")
    anchor_blocks = extract_anchor_blocks(context, answers, query, tokenizer, embedder)
    if not anchor_blocks:
        # fallback: use first 256 tokens of context as single anchor
        enc = tokenizer(context, add_special_tokens=False, truncation=True, max_length=256)
        fallback_text = tokenizer.decode(enc["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        anchor_blocks = [fallback_text]

    # Calculate dynamic question length limit (original + 20 tokens)
    orig_q_tokens = len(tokenizer.encode(seed_item.get("input", ""), add_special_tokens=False))
    max_q_tokens_allowed = orig_q_tokens + 20

    masked_ctx = build_masked_context(anchor_blocks)

    prompt_template = FEW_SHOT_JSON
    prompt_template += f"You are given colored anchor segments (in <ANCHOR_i> tags) from a longer document.\n"
    prompt_template += f"The original question was: \"{query}\".\n"
    prompt_template += f"Generate up to {n} NEW question-answer pairs in JSON array format.\n"
    prompt_template += (
        "Each object must contain: \n"
        "  \"q\": a question answerable ONLY from the colored text, distinct from previous questions, "
        f"and no longer than {max_q_tokens_allowed} tokens.\n"
        "  \"a\": a concise answer (<= 100 tokens) copied verbatim from the colored text.\n"
    )
    prompt_template += "No two questions may be duplicate. Only output valid JSON, nothing else.\n\n"
    prompt_template += masked_ctx

    return prompt_template, anchor_blocks

###############################################
# Validate QA pair
###############################################

# Add stopword list for filtering
STOPWORDS = set([
    'the', 'a', 'an', 'he', 'she', 'it', 'they', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'and', 'or', 'but', 'is', 'was', 'are', 'were',
    'this', 'that', 'these', 'those', 'his', 'her', 'their', 'its', 'by', 'as', 'from', 'be', 'been', 'has', 'had', 'have', 'will', 'would', 'can', 'could', 'should', 'may', 'might', 'do', 'does', 'did', 'not', 'so', 'if', 'then', 'than', 'also', 'such', 'which', 'who', 'whom', 'whose', 'what', 'when', 'where', 'why', 'how', 'about', 'into', 'over', 'after', 'before', 'between', 'under', 'again', 'further', 'once', 'because', 'while', 'during', 'above', 'below', 'up', 'down', 'out', 'off', 'just', 'now', 'only', 'own', 'same', 'too', 'very', 's', 't', 'can', 'will', 'don', 'should', 'll', 're', 've', 'y', 'd', 'm'
])

# Helper: Remove stopwords from a string

def remove_stopwords(text):
    return ' '.join([w for w in text.lower().split() if w not in STOPWORDS])

# Helper: For short answers (numbers, pronouns), expand to containing sentence

def expand_to_sentence(context, answer):
    idx = context.lower().find(answer.lower())
    if idx == -1:
        return answer
    # Find sentence boundaries
    left = context.rfind('.', 0, idx)
    right = context.find('.', idx)
    if left == -1: left = 0
    else: left += 1
    if right == -1: right = len(context)
    return context[left:right].strip()

# Hybrid fuzzy/semantic answer-in-anchor filter

def _answer_in_anchors_fuzzy(answer, anchor_blocks, embedder, threshold=0.7):
    answer_clean = remove_stopwords(answer)
    if not answer_clean:
        return False
    # Exact substring
    for blk in anchor_blocks:
        if answer_clean in blk.lower():
            return True
    # For short answers (<= 4 tokens), expand to sentence
    if len(answer_clean.split()) <= 4:
        expanded = expand_to_sentence(' '.join(anchor_blocks), answer)
        answer_clean = expanded
    # Batch embed for efficiency
    answer_emb = embedder.encode([answer_clean], convert_to_tensor=True)
    anchor_embs = embedder.encode(anchor_blocks, convert_to_tensor=True)
    sims = util.cos_sim(answer_emb, anchor_embs)[0]
    if torch.any(sims > threshold):
        return True
    return False

def _token_len(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

###############################################
# FastRAG-style MMR re-ranker for diversity
###############################################

def mmr_select(qas: List[Dict[str, str]], embedder, k: int = 6, lambda_coeff: float = 0.5):
    """Return up to k QA dicts using Max-Marginal-Relevance on the question embeddings."""
    if len(qas) <= k:
        return qas
    questions = [qa.get("q", "") for qa in qas]
    embeds = embedder.encode(questions, convert_to_tensor=True)
    if embeds.shape[0] == 0:
        return []
    selected_idx = [int(torch.argmax(torch.norm(embeds, dim=1)))]  # start with the longest vector (proxy info)
    candidates = set(range(len(qas))) - set(selected_idx)
    while len(selected_idx) < k and candidates:
        best_score = -1
        best_idx = None
        for idx in candidates:
            sim_to_query = torch.max(util.cos_sim(embeds[idx], embeds)).item()
            sim_to_selected = torch.max(util.cos_sim(embeds[idx], embeds[selected_idx])).item()
            score = lambda_coeff * sim_to_query - (1 - lambda_coeff) * sim_to_selected
            if score > best_score:
                best_score = score
                best_idx = idx
        selected_idx.append(best_idx)
        candidates.remove(best_idx)
    return [qas[i] for i in selected_idx]

# Mini-QA pipeline (loaded once, GPU)
mini_qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=0)

# Tolerant fuzzy match for QA validation

def tolerant_fuzzy_match(pred, gold, embedder, context, gold_context, threshold=0.6):
    pred, gold = pred.lower().strip(), gold.lower().strip()
    if not pred or not gold:
        return False
    if pred in gold or gold in pred:
        return True
    pred_tokens, gold_tokens = set(pred.split()), set(gold.split())
    if len(pred_tokens & gold_tokens) / max(1, len(gold_tokens)) > 0.3:
        return True
    # Embedding similarity (sentence level)
    pred_sent = expand_to_sentence(context, pred)
    gold_sent = expand_to_sentence(gold_context, gold)
    pred_emb = embedder.encode([pred_sent], convert_to_tensor=True)
    gold_emb = embedder.encode([gold_sent], convert_to_tensor=True)
    sim = torch.nn.functional.cosine_similarity(pred_emb, gold_emb).item()
    return sim > threshold

# Soft anchor validation with mini-QA

def soft_validate_anchors_with_miniqa(question, gold_answer, anchor_blocks, embedder, gold_context):
    qa_inputs = [{"question": question, "context": blk} for blk in anchor_blocks]
    results = mini_qa(qa_inputs)
    for blk, res in zip(anchor_blocks, results):
        pred = res['answer'].strip()
        if tolerant_fuzzy_match(pred, gold_answer, embedder, blk, gold_context):
            return True
    return False

# ---------------------------------------------------------------------
# 3.  New-item constructor
# ---------------------------------------------------------------------
def make_item_from_pair(seed: Dict[str, Any], q: str, a: str) -> Dict[str, Any]:
    """Return a new datapoint that keeps both contexts unchanged."""
    return {
        "input": q,
        "answers": a,
        "context_A": seed["context_A"],
        "context_B": seed["context_B"],
        "question_type": seed["question_type"],
        "language": seed["language"],
        "use_cot": seed.get("use_cot", True),
        "_id": shortuuid.uuid(),
        "seed_id": seed["_id"],         # traceability
    }

# ---------------------------------------------------------------------
# 4.  Generation routine
# ---------------------------------------------------------------------
def generate_qa_data(seed_path: str = "data/data_seed.json",
                     out_dir: str = "data",
                     qa_per_seed: int = 4,
                     similarity_threshold: float = 0.8,
                     batch_size: int = 8,
                     max_new_items: Optional[int] = None):
    llm = get_llm()
    tokenizer = get_tokenizer()
    embedder = get_embedder()

    # --- load seeds ---
    with open(seed_path, "r", encoding="utf-8") as f:
        seeds = json.load(f)
    print(f"Loaded {len(seeds)} seed examples")

    regen_path = os.path.join(out_dir, "regen.json")
    os.makedirs(out_dir, exist_ok=True)
    regen_data: List[Dict[str, Any]] = []
    if os.path.isfile(regen_path):
        regen_data = json.load(open(regen_path, encoding="utf-8"))
        print(f"Loaded {len(regen_data)} existing items from regen.json")

    # build embedding set of existing QUESTIONS (inputs)
    exist_inputs = [d["input"] for d in regen_data]
    exist_embs = embedder.encode(exist_inputs, convert_to_tensor=True) if exist_inputs else None

    prompts, seed_refs, prompt_anchors = [], [], []
    for s in seeds:
        for k in range(qa_per_seed):
            prompt, anchors = build_prompt(s, qa_per_seed, k, tokenizer, embedder)
            prompts.append(prompt)
            seed_refs.append(s)
            prompt_anchors.append(anchors)

    sampling_args = SamplingParams(
        temperature=0.5, top_p=0.8, top_k=40, max_tokens=512
    )

    new_items: List[Dict[str, Any]] = []
    for i in tqdm.tqdm(range(0, len(prompts), batch_size), desc="LLM batches"):
        batch_prompts = prompts[i:i+batch_size]
        outputs = llm.generate(batch_prompts, sampling_args)
        for local_idx, (prompt_out, seed, anchors) in enumerate(zip(outputs, seed_refs[i:i+batch_size], prompt_anchors[i:i+batch_size])):
            try:
                text = prompt_out.outputs[0].text.strip()
                qa_list = json.loads(text)
                if not isinstance(qa_list, list):
                    raise ValueError("LLM output is not a list")
            except Exception:
                continue

            # Re-rank with MMR for semantic diversity
            qa_list = mmr_select(qa_list, embedder, k=qa_per_seed*3)

            # maintain bucket counts per anchor id
            anchor_bucket_counts = defaultdict(int)
            for qa in qa_list:
                q = qa.get("q", "").strip()
                a = qa.get("a", "").strip()
                if not q or not a:
                    continue
                # Basic sanity filters
                if _token_len(a, tokenizer) > 100:
                    continue
                if any(tok in a for tok in ["http", "`", "<", "</", "\\"]):
                    continue
                # Hard cap question length to original + 20 tokens
                orig_len = _token_len(seed["input"], tokenizer)
                if _token_len(q, tokenizer) > orig_len + 20:
                    continue
                # Soft mini-QA validation (not a hard filter)
                try:
                    if not soft_validate_anchors_with_miniqa(q, a, anchors, embedder, seed["context_A"]):
                        pass  # allow through, just log or optionally skip
                except Exception as e:
                    pass  # fail open for speed/robustness
                if not _answer_in_anchors_fuzzy(a, anchors, embedder, threshold=0.7):
                    continue

                # duplicate filtering
                if exist_embs is not None and len(exist_embs) > 0:
                    q_emb = embedder.encode(q, convert_to_tensor=True)
                    score = util.cos_sim(q_emb, exist_embs).max().item()
                    if score >= similarity_threshold:
                        continue
                    exist_inputs.append(q)
                    exist_embs = torch.cat([exist_embs, q_emb.unsqueeze(0)], dim=0)
                # anchor bucket limit (e.g., 4 per anchor)
                anchor_idx = next((idx for idx, blk in enumerate(anchors) if a.lower() in blk.lower()), 0)
                if anchor_bucket_counts[anchor_idx] >= qa_per_seed * 2:
                    continue
                anchor_bucket_counts[anchor_idx] += 1

                item = make_item_from_pair(seed, q, a)
                new_items.append(item)
                if max_new_items and len(new_items) >= max_new_items:
                    break
            if max_new_items and len(new_items) >= max_new_items:
                break
        if max_new_items and len(new_items) >= max_new_items:
            print("Reached cap; stopping generation.")
            break

    print(f"Accepted {len(new_items)} new (q,a) pairs")
    regen_data.extend(new_items)
    with open(regen_path, "w", encoding="utf-8") as f:
        json.dump(regen_data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(regen_data)} total items to {regen_path}")

# ---------------------------------------------------------------------
# 5.  CLI
# ---------------------------------------------------------------------
def main(**kwargs):
    generate_qa_data(**kwargs)

# Backward-compat alias so existing scripts importing `from generate_qa_data import generate_data` work
generate_data = generate_qa_data

if __name__ == "__main__":
    fire.Fire(main)
