from copy import deepcopy
from functools import partial

import pandas as pd
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from eedi import (
    KNOWLEDGE_TYPE_GENK,
    KNOWLEDGE_TYPE_NONE,
    KNOWLEDGE_TYPE_RAG,
    KNOWLEDGE_TYPE_TOT,
    MODEL_ID_LLAMA31_8B,
    MODEL_ID_LLAMA32_3B,
    MODEL_ID_QWEN25_7B,
)
from eedi.prompts import (
    llama_base_task_prompt,
    llama_genk_knowledge_prompt,
    llama_knowledge_task_prompt,
    llama_tot_knowledge_prompt,
    qwen_base_task_prompt,
    qwen_genk_knowledge_prompt,
    qwen_knowledge_task_prompt,
    qwen_tot_knowledge_prompt,
)


def llama_remove_prompt(record: pd.Series, resp_key: str) -> str:
    l = record[resp_key].index("<|start_header_id|>assistant<|end_header_id|>") + 45
    value = record[resp_key][l:].strip()
    if value == "":
        value = "No Misconception Found"
    return value


def qwen_remove_prompt(record: pd.Series, resp_key: str) -> str:
    l = record[resp_key].index("<|im_start|>assistant") + 21
    value = record[resp_key][l:].strip()
    if value == "":
        value = "No Misconception Found"
    return value


PROMPT_REMOVER_MAP = {
    MODEL_ID_LLAMA32_3B: llama_remove_prompt,
    MODEL_ID_LLAMA31_8B: llama_remove_prompt,
    MODEL_ID_QWEN25_7B: qwen_remove_prompt,
}


def apply_task_template(
    record: pd.Series,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt.format(
                ConstructName=record["ConstructName"],
                SubjectName=record["SubjectName"],
                Question=record["QuestionText"],
                IncorrectAnswer=record[f"AnswerText"],
                CorrectAnswer=record[f"CorrectAnswerText"],
            ),
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def apply_task_with_knowledge_template(
    record: pd.Series,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt.format(
                Knowledge=record["Knowledge"],
                ConstructName=record["ConstructName"],
                SubjectName=record["SubjectName"],
                Question=record["QuestionText"],
                IncorrectAnswer=record[f"AnswerText"],
                CorrectAnswer=record[f"CorrectAnswerText"],
            ),
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


LLAMA_BASE_TASK = partial(
    apply_task_template,
    prompt=llama_base_task_prompt,
)
LLAMA_KNOWLEDGE_TASK = partial(
    apply_task_with_knowledge_template,
    prompt=llama_knowledge_task_prompt,
)
LLAMA_TASK_TEMPLATES = {
    KNOWLEDGE_TYPE_NONE: LLAMA_BASE_TASK,
    KNOWLEDGE_TYPE_GENK: LLAMA_KNOWLEDGE_TASK,
    KNOWLEDGE_TYPE_TOT: LLAMA_KNOWLEDGE_TASK,
    KNOWLEDGE_TYPE_RAG: LLAMA_KNOWLEDGE_TASK,
}
QWEN_BASE_TASK = partial(
    apply_task_template,
    prompt=qwen_base_task_prompt,
)
QWEN_KNOWLEDGE_TASK = partial(
    apply_task_with_knowledge_template,
    prompt=qwen_knowledge_task_prompt,
)
QWEN_TASK_TEMPLATES = {
    KNOWLEDGE_TYPE_NONE: QWEN_BASE_TASK,
    KNOWLEDGE_TYPE_GENK: QWEN_KNOWLEDGE_TASK,
    KNOWLEDGE_TYPE_TOT: QWEN_KNOWLEDGE_TASK,
    KNOWLEDGE_TYPE_RAG: QWEN_KNOWLEDGE_TASK,
}
TASK_TEMPLATE_FUNC_MAP = {
    MODEL_ID_LLAMA32_3B: LLAMA_TASK_TEMPLATES,
    MODEL_ID_LLAMA31_8B: LLAMA_TASK_TEMPLATES,
    MODEL_ID_QWEN25_7B: QWEN_TASK_TEMPLATES,
}


def apply_knowledge_template(
    record: pd.Series,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt.format(
                ConstructName=record["ConstructName"],
                SubjectName=record["SubjectName"],
            ),
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


LLAMA_KNOWLEDGE_TEMPLATES = {
    KNOWLEDGE_TYPE_NONE: None,
    KNOWLEDGE_TYPE_GENK: partial(
        apply_knowledge_template,
        prompt=llama_genk_knowledge_prompt,
    ),
    KNOWLEDGE_TYPE_TOT: partial(
        apply_knowledge_template,
        prompt=llama_tot_knowledge_prompt,
    ),
    KNOWLEDGE_TYPE_RAG: None,
}
QWEN_KNOWLEDGE_TEMPLATES = {
    KNOWLEDGE_TYPE_NONE: None,
    KNOWLEDGE_TYPE_GENK: partial(
        apply_knowledge_template,
        prompt=qwen_genk_knowledge_prompt,
    ),
    KNOWLEDGE_TYPE_TOT: partial(
        apply_knowledge_template,
        prompt=qwen_tot_knowledge_prompt,
    ),
    KNOWLEDGE_TYPE_RAG: None,
}
KNOWLEDGE_TEMPLATE_FUNC_MAP = {
    MODEL_ID_LLAMA32_3B: LLAMA_KNOWLEDGE_TEMPLATES,
    MODEL_ID_LLAMA31_8B: LLAMA_KNOWLEDGE_TEMPLATES,
    MODEL_ID_QWEN25_7B: QWEN_KNOWLEDGE_TEMPLATES,
}
