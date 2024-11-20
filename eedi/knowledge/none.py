from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from transformers.generation import GenerationMixin
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def enhance_with_knowledge(
    llm: GenerationMixin,
    llm_tokenizer: PreTrainedTokenizerFast,
    encoder: SentenceTransformer,
    df_xy: DataFrame,
    batch_size: int,
    disable_tqdm: bool,
) -> DataFrame:
    """should return dataframe with new knowledge column

    :param llm: _description_
    :type llm: GenerationMixin
    :param llm_tokenizer: _description_
    :type llm_tokenizer: PreTrainedTokenizerFast
    :param encoder: _description_
    :type encoder: SentenceTransformer
    :param df_xy: _description_
    :type df_xy: DataFrame
    :param batch_size: _description_
    :type batch_size: int
    :param disable_tqdm: _description_
    :type disable_tqdm: bool
    :return: _description_
    :rtype: DataFrame
    """
    return df_xy
