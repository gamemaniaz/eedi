from typing import Callable

from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from transformers.generation import GenerationMixin
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tqdm import tqdm
from eedi.utils import get_device
import torch

class ThoughtNode:
    def __init__(self, thought=None, result=None, children=None):
        self.thought = thought
        self.result = result
        self.children = children or []


class TreeOfThought:
    def __init__(self, root_prompt, max_iterations=3, max_tokens=4096):
        self.root = ThoughtNode(root_prompt)
        self.max_iterations = max_iterations
        self.current_thoughts = [self.root]
        self.max_tokens = max_tokens


explain_correct_option = "Please succinctly explain the correction option:"
rate_correct_option = "Please reply only with the best explanation, in the format 'explanation X' and nothing else "
# misconception_prompt = f"If a student chose option {option}, please explain the misconception the student might have"
rate_correct_misconception = "Please reply only with the best misconception, in the format 'misconception X' and nothing else "


def gen(prompt: str, model: GenerationMixin, tokenizer: PreTrainedTokenizerFast) -> str:
    tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(get_device())
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            tokens.input_ids,
            max_new_tokens=1024,
            num_return_sequences=1,
            attention_mask=tokens.attention_mask,
        )
    return tokenizer.decode(output_ids)


initial_prompt_template = """Question: {Question}
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
    root_node = ThoughtNode()


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
    return df_xy_enhanced
