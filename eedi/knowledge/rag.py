from typing import Callable

from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from transformers.generation import GenerationMixin
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tqdm import tqdm
from eedi import RESULTS_DIR
from eedi.utils import save_df


prompt_template = """Question: {Question}
Incorrect Answer: {IncorrectAnswer}
Correct Answer: {CorrectAnswer}
Construct Name: {ConstructName}
Subject Name: {SubjectName}"""


def generate_knowledge(
    *,
    llm: GenerationMixin,
    llm_tokenizer: PreTrainedTokenizerFast,
    Question: str,
    IncorrectAnswer: str,
    CorrectAnswer: str,
    ConstructName: str,
    SubjectName: str,
) -> str:
    question_prompt = prompt_template.format(
        Question=Question,
        IncorrectAnswer=IncorrectAnswer,
        CorrectAnswer=CorrectAnswer,
        ConstructName=ConstructName,
        SubjectName=SubjectName,
    )
    return ""



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
    df_xy_enhanced = df_xy.copy(deep=True)
    knowledges = []
    for _, row in tqdm(df_xy_enhanced.iterrows(), desc="gentot"):
        knowledges.append(generate_knowledge(
            llm,
            llm_tokenizer,
            row["Question"],
            row["IncorrectAnswer"],
            row["CorrectAnswer"],
            row["ConstructName"],
            row["SubjectName"],
        ))
    df_xy_enhanced["Knowledge"] = knowledges
    save_df(df_xy_enhanced, RESULTS_DIR, run_id, "df_xy_enhanced.parquet")
    return df_xy_enhanced
