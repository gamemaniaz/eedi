from typing import Callable

from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from transformers.generation import GenerationMixin
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def enhance_with_knowledge(
    *,
    llm: GenerationMixin,
    llm_tokenizer: PreTrainedTokenizerFast,
    encoder: SentenceTransformer,
    df_xy: DataFrame,
    batch_size: int,
    disable_tqdm: bool,
    knowledge_template_func: Callable,
    remove_prompt_func: Callable,
    run_id: str,
) -> DataFrame:
    """should return dataframe with new knowledge column"""
    return
