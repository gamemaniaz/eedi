import re
from pathlib import Path

import pandas as pd
import torch
from ingestion import ingest_data
from preprocess import preprocess_data
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def main():
    df_train, df_test, df_sample_submission, df_misconception_mapping = ingest_data()
    # preprocess_data(df_train, df_test, df_sample_submission, df_misconception_mapping)
