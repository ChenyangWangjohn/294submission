# create the initial seed data for later generation
# The seed data is randomly picked from the original dataset
# the seed longbench_C.json will be randomly sampled to create the seed data

import os
import json
import random
import shortuuid

# Path to your combined A/B dataset
DATA_PATH = "data/longbench_C.json"

# Subsample ratio (e.g., 20%)
SUBSAMPLE_SPLIT = 0.2

def main():
    # 1) Ensure output directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Ensure input data exists and seed random for reproducibility
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Input dataset '{DATA_PATH}' not found. Please run data_prep.py first.")
    
    # Set random seed for reproducibility
    random.seed(42)

    # 2) Load entire dataset (single JSON array)
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f)  # expecting a list of dicts

    # 3) Randomly sample
    subsample_size = int(SUBSAMPLE_SPLIT * len(all_data))
    subsampled_data = random.sample(all_data, subsample_size)

    # 4) Reformat to desired structure
    formatted_data = []
    for dp in subsampled_data:
        formatted_data.append({
            "input": dp.get("input", ""),
            "context_A": dp.get("context_A", ""),
            "context_B": dp.get("context_B", ""),
            "answers": dp.get("answers", ""),
            "question_type": dp.get("question_type", ""),
            "language": dp.get("language", ""),
            "use_cot": dp.get("use_cot", True),
            "_id": dp.get("_id") or shortuuid.uuid(),
        })

    # 5) Save as a single JSON array (seed data)
    output_file = "data/data_seed.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(formatted_data)} seed examples to '{output_file}'")

if __name__ == "__main__":
    main()