import json
import os
from typing import Dict, List
import pandas as pd
from loguru import logger
from tqdm import tqdm


# clear contents of log file
with open("logs/token_stats.log", "w") as f:
    pass

# set log file
logger.add("logs/token_stats.log", rotation="500 MB")


def extract_token_counts(strategy: str, split: str) -> Dict[str, List[int]]:
    """
    Extracts token counts from data files.
    """
    assert strategy in ["self_ask", "direct"]
    assert split in ["train", "dev", "test"]

    # read data file
    DATA_DIR = "data/FinetuningData/"
    data_file = os.path.join(DATA_DIR, f'{strategy}_{split}.json')
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # count tokens
    prompt_token_counts = []
    target_token_counts = []
    total_token_counts = []
    for record in tqdm(data, desc=f"Extracting token counts for {strategy}-{split} split"):
        prompt_token_counts.append(record['num_prompt_tokens'])
        target_token_counts.append(record['num_target_tokens'])
        total_token_counts.append(record['num_tokens'])
    return {"prompt_token_counts": prompt_token_counts,
            "target_token_counts": target_token_counts,
            "total_token_counts": total_token_counts}


def summarize_token_counts(token_counts: Dict[str, List[int]]) -> pd.DataFrame:
    """
    Summarizes token counts.
    """
    # convert token_counts to pandas dataframe
    df = pd.DataFrame(token_counts)
    # create summary dataframe
    summary = df.describe()
    # add sum of columns
    summary.loc['sum'] = df.sum()
    return summary.round(2)


def main():
    for strategy in ["self_ask", "direct"]:
        for split in ["train", "dev"]:
            counts = extract_token_counts(strategy, split)
            # summarize token counts
            stats = summarize_token_counts(counts)
            logger.info(f"Token stats for {strategy}-{split} split:\n{stats.to_string()}")


if __name__ == "__main__":
    main()