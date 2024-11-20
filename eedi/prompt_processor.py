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
    llama_genk_task_prompt,
    llama_rag_task_prompt,
    llama_tot_knowledge_prompt,
    llama_tot_task_prompt,
    qwen_base_task_prompt,
    qwen_genk_knowledge_prompt,
    qwen_genk_task_prompt,
    qwen_rag_task_prompt,
    qwen_tot_knowledge_prompt,
    qwen_tot_task_prompt,
)


def llama_remove_prompt(record: pd.Series):
    l = record["FullResponse"].index("<|start_header_id|>assistant<|end_header_id|>") + 45
    value = record["FullResponse"][l:].strip()
    if value == "":
        value = "No Misconception Found"
    return value


def qwen_remove_prompt(record: pd.Series, resp_key: str):
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


def apply_base_template(record: pd.Series, tokenizer: PreTrainedTokenizerFast, prompt: str):
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


def apply_knowledge_template(record: pd.Series, tokenizer: PreTrainedTokenizerFast, prompt: str):
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


llama_templates = {
    KNOWLEDGE_TYPE_NONE: partial(
        func=apply_base_template,
        prompt=llama_base_task_prompt,
    ),
    KNOWLEDGE_TYPE_GENK: partial(
        func=apply_knowledge_template,
        prompt=llama_genk_task_prompt,
    ),
    KNOWLEDGE_TYPE_TOT: partial(
        func=apply_knowledge_template,
        prompt=llama_tot_task_prompt,
    ),
    KNOWLEDGE_TYPE_RAG: partial(
        func=apply_knowledge_template,
        prompt=llama_rag_task_prompt,
    ),
}

qwen_templates = {
    KNOWLEDGE_TYPE_NONE: partial(
        func=apply_base_template,
        prompt=qwen_base_task_prompt,
    ),
    KNOWLEDGE_TYPE_GENK: partial(
        func=apply_knowledge_template,
        prompt=qwen_genk_task_prompt,
    ),
    KNOWLEDGE_TYPE_TOT: partial(
        func=apply_knowledge_template,
        prompt=qwen_tot_task_prompt,
    ),
    KNOWLEDGE_TYPE_RAG: partial(
        func=apply_knowledge_template,
        prompt=qwen_rag_task_prompt,
    ),
}

TEMPLATE_FUNC_MAP = {
    MODEL_ID_LLAMA32_3B: llama_templates,
    MODEL_ID_LLAMA31_8B: llama_templates,
    MODEL_ID_QWEN25_7B: qwen_templates,
}
