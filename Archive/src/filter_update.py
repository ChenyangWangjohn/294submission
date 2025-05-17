"""
filter_update.py

Takes in a judge output file like `data/judge.jsonl`, 
finds all seed_ids that failed, 
filters the original seed_data to only those items, 
and calls generate_data(...) to produce new data for them.

Usage from another script:
  from filter import filter_and_generate
  filter_and_generate(
      judge_file="data/judge.jsonl",
      seed_data_path="data/data_seed.json",
      output_dir="data"
  )
"""

import os
import json
import shortuuid

# If your generate_data code is in the same project,
# import it. Example:
from generate_qa_data import generate_data  # switched to new QA generation pipeline

######################################
# Hyperparameters for filtering + generation
######################################
JUDGE_FILE = "data/judge.jsonl"
SEED_DATA_PATH = "data/data_seed.json"
OUTPUT_DIR = "data"  # same directory as your existing 'regen.json'
INSTRUCTIONS_PER_SEED_TASK = 4
# Threshold for duplicate filtering in generate_data
GENERATION_SIMILARITY_THRESHOLD = 0.9 # Updated from ROUGE to Cosine Similarity
# GENERATION_WORKERS removed as it's deprecated in generate_data

def filter_and_generate(
    judge_file: str = JUDGE_FILE,
    seed_data_path: str = SEED_DATA_PATH,
    output_dir: str = OUTPUT_DIR,
    instructions_per_seed_task: int = INSTRUCTIONS_PER_SEED_TASK,
    similarity_threshold: float = GENERATION_SIMILARITY_THRESHOLD # Renamed from rouge_score_threshold
    # generation_workers removed
):
    """
    1) Load the judge results from `judge_file` (line-by-line JSON).
    2) Collect all seed_ids that fail the threshold or have 'wrong' criteria.
    3) Load the full seed data from `seed_data_path`.
    4) Filter only the seeds with those failing seed_ids.
    5) Create a temp file with that partial subset.
    6) Call generate_data(...) on that subset, preserving and updating the old regen.json.
    """

    # If judge file does not exist (e.g. first run), fall back to generating data for *all* seeds
    if not os.path.exists(judge_file):
        print(f"[filter_and_generate] Judge file '{judge_file}' not found. Assuming first pass – generating data for the full seed set.")
        generate_data(
            seed_tasks_path=seed_data_path,
            output_dir=output_dir,
            instructions_per_seed_task=instructions_per_seed_task,
            similarity_threshold=similarity_threshold, # Renamed argument
            # generation_workers removed
        )
        print("[filter_and_generate] Initial regen.json has been created.")
        return

    if not os.path.exists(seed_data_path):
        raise FileNotFoundError(f"Seed data {seed_data_path} not found.")

    # Ensure regen.json exists – if it's missing we build it once from the full seed set
    regen_path = os.path.join(output_dir, "regen.json")
    if not os.path.exists(regen_path):
        print(f"[filter_and_generate] regen file '{regen_path}' not found. Creating an empty one for augmented data...")
        os.makedirs(output_dir, exist_ok=True)
        with open(regen_path, "w", encoding="utf-8") as rf:
            json.dump([], rf)
        print("[filter_and_generate] Empty regen.json created.")

    # 1) Read the judge results, collect failing seed_ids
    failing_seed_ids = set()
    lines_count = 0
    try:
        with open(judge_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lines_count += 1
                try:
                    dp = json.loads(line)
                    # We assume any line here indicates a fail or threshold not met
                    seed_id = dp.get("seed_id", "")
                    if seed_id:
                        failing_seed_ids.add(seed_id)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {judge_file}: {line}")
    except FileNotFoundError:
        # This case is handled at the beginning, but added for robustness
        print(f"Error: Judge file {judge_file} not found unexpectedly.")
        return

    print(f"[filter_and_generate] Found {len(failing_seed_ids)} failing seed_ids among {lines_count} judge lines.")

    if not failing_seed_ids:
        print("[filter_and_generate] No failing seed_ids found. Nothing to generate.")
        return

    # 2) Load the full seed data
    try:
        with open(seed_data_path, "r", encoding="utf-8") as f:
            full_seed_data = json.load(f)
    except FileNotFoundError:
         print(f"Error: Seed data file {seed_data_path} not found.")
         return
    except json.JSONDecodeError:
        print(f"Error: Seed data file {seed_data_path} is corrupted.")
        return

    print(f"[filter_and_generate] Loaded {len(full_seed_data)} seeds from {seed_data_path}.")

    # 3) Filter only seeds with seed_id in failing_seed_ids
    partial_seeds = []
    for item in full_seed_data:
        sid = item.get("_id", "")
        if sid in failing_seed_ids:
            # Ensure the item gets a unique ID if it doesn't have one,
            # though seed data should ideally have them.
            new_item = {
                **item,
                "_id": item.get("_id") or shortuuid.uuid(),
            }
            partial_seeds.append(new_item)

    print(f"[filter_and_generate] Filtered seeds: {len(partial_seeds)} items to augment.")

    if not partial_seeds:
        print("[filter_and_generate] No matched seeds for the failing IDs.")
        return

    # 4) Create a temporary JSON file for these partial seeds
    tmp_seed_file = os.path.join(output_dir, "tmp_failing_seeds.json")
    try:
        with open(tmp_seed_file, "w", encoding="utf-8") as f:
            json.dump(partial_seeds, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Error writing temporary seed file {tmp_seed_file}: {e}")
        return

    print(f"[filter_and_generate] Wrote partial seeds to {tmp_seed_file}.")

    # 5) Now call generate_data(...) with this partial seed file.
    #    This function presumably loads 'regen.json' if it exists,
    #    appends new items, and saves it again. That means we keep old data.

    generate_data(
        seed_path=tmp_seed_file,       # the partial set
        out_dir=output_dir,               # same dir as old regen
        qa_per_seed=instructions_per_seed_task,
        similarity_threshold=similarity_threshold, # Renamed argument
    )

    # Clean up temporary file (optional)
    try:
        os.remove(tmp_seed_file)
        print(f"[filter_and_generate] Removed temporary file {tmp_seed_file}.")
    except OSError as e:
        print(f"Warning: Could not remove temporary file {tmp_seed_file}: {e}")

    # Done. generate_data should have updated 'regen.json'
    print("[filter_and_generate] Completed. 'regen.json' updated with new data.")


if __name__ == "__main__":
    # Example CLI usage (optional):
    import argparse
    parser = argparse.ArgumentParser(
        description="Filter seed data based on judge results and generate new data for failing items."
    )
    parser.add_argument("--judge_file", type=str, default=JUDGE_FILE,
                        help=f"Path to the judge results file (JSONL format). Default: {JUDGE_FILE}")
    parser.add_argument("--seed_data_path", type=str, default=SEED_DATA_PATH,
                        help=f"Path to the original seed data file (JSON format). Default: {SEED_DATA_PATH}")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help=f"Directory to load/save 'regen.json' and write temporary files. Default: {OUTPUT_DIR}")
    parser.add_argument("--instructions_per_seed_task", type=int, default=INSTRUCTIONS_PER_SEED_TASK,
                        help=f"Number of new data samples to generate per failing seed item. Default: {INSTRUCTIONS_PER_SEED_TASK}")
    parser.add_argument("--similarity_threshold", type=float, default=GENERATION_SIMILARITY_THRESHOLD,
                        help=f"Cosine similarity threshold for filtering generated data duplicates. Default: {GENERATION_SIMILARITY_THRESHOLD}")
    # generation_workers argument removed
    args = parser.parse_args()

    filter_and_generate(
        judge_file=args.judge_file,
        seed_data_path=args.seed_data_path,
        output_dir=args.output_dir,
        instructions_per_seed_task=args.instructions_per_seed_task,
        similarity_threshold=args.similarity_threshold, # Use renamed argument
        # generation_workers removed
    )
