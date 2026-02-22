from pathlib import Path

import pandas as pd

from modules.logging_colors import logger


def load_past_evaluations():
    if Path("user_data/logs/evaluations.csv").exists():
        df = pd.read_csv(Path("user_data/logs/evaluations.csv"), dtype=str)
        df["Perplexity"] = pd.to_numeric(df["Perplexity"])
        return df
    return pd.DataFrame(columns=["Model", "LoRAs", "Dataset", "Perplexity", "stride", "max_length", "Date", "Comment"])


past_evaluations = load_past_evaluations()


def save_past_evaluations(df):
    global past_evaluations
    past_evaluations = df
    filepath = Path("user_data/logs/evaluations.csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def calculate_perplexity(models, input_dataset, stride, _max_length):
    """
    Based on:
    https://huggingface.co/docs/transformers/perplexity#calculating-ppl-with-fixedlength-models
    """
    logger.error("Perplexity evaluation is not implemented for the llama.cpp loader.")
    raise ValueError


def generate_markdown_table():
    sorted_df = past_evaluations.sort_values(by=["Dataset", "stride", "Perplexity", "Date"])
    return sorted_df
