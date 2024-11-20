from typing import Callable

from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from transformers.generation import GenerationMixin
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class ThoughtNode:
    def __init__(self, thought=None, result=None, children=None):
        self.thought = thought
        self.result = result
        self.children = children or []


class TreeOfThought:
    def __init__(self, root_prompt, max_iterations=3, max_tokens=250):
        self.root = ThoughtNode(root_prompt)
        self.max_iterations = max_iterations
        self.current_thoughts = [self.root]
        self.max_tokens = max_tokens


explain_correct_option = "Please succinctly explain the correction option:"
rate_correct_option = "Please reply only with the best explanation, in the format 'explanation X' and nothing else "
# misconception_prompt = f"If a student chose option {option}, please explain the misconception the student might have"
rate_correct_misconception = "Please reply only with the best misconception, in the format 'misconception X' and nothing else "


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

    # TODO

    return
