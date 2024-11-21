from __future__ import annotations
from typing import Callable
import re

from pandas import DataFrame, Series
from sentence_transformers import SentenceTransformer
from transformers.generation import GenerationMixin
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tqdm import tqdm
from eedi.utils import get_device
import torch
from eedi import RESULTS_DIR
from eedi.utils import save_df


prompt_template = """Question: {Question}
Incorrect Answer: {IncorrectAnswer}
Correct Answer: {CorrectAnswer}
Construct Name: {ConstructName}
Subject Name: {SubjectName}"""
ask_misconception = "Given the above details, explain the misconception the student has and wrap it in these tags <misconception></misconception>"
rate_prompt = "Given the above details and the misconceptions in tags <misconception></misconception>, choose only the best misconception and put it in <bestmisconception></bestmisconception>"


def gen(prompt: str, model: GenerationMixin, tokenizer: PreTrainedTokenizerFast) -> str:
    tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(get_device())
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            tokens.input_ids,
            max_new_tokens=512,
            num_return_sequences=1,
            attention_mask=tokens.attention_mask,
            temperature=0.7,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def get_miscon(text: str) -> str:
    miscons = re.findall(r"<misconception>(?s:.*?)</misconception>", text)
    miscons = [x.strip().replace("<misconception>", "").replace("</misconception>", "") for x in miscons]
    return " ".join(miscons).strip()


def get_bestmiscon(text: str) -> str:
    miscons = re.findall(r"<bestmisconception>(?s:.*?)</bestmisconception>", text)
    miscons = [x.strip().replace("<bestmisconception>", "").replace("</bestmisconception>", "") for x in miscons]
    return " ".join(miscons).strip()


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
    depth = 3
    fanout = 3

    selected_miscon = ""
    for _ in range(depth):
        miscons = []
        if selected_miscon != "":
            curr_question_prompt = question_prompt + f"\n\n<current-misconception>{selected_miscon}</current-misconception>"
        for _ in range(fanout):
            question = curr_question_prompt + f"\n\n{ask_misconception}"
            response = gen(question, llm, llm_tokenizer)[len(question):]
            miscon = get_miscon(response)
            miscons.append(miscon)
        if not miscons:
            break
        compiled_miscon_prompt = curr_question_prompt + "\n"
        for miscon in miscons:
            compiled_miscon_prompt += f"\n<misconception>{miscon}</misconception>"
        rate_prompt_with_given_miscon = compiled_miscon_prompt + f"\n\n{rate_prompt}"
        best_response = gen(rate_prompt_with_given_miscon, llm, llm_tokenizer)[len(rate_prompt_with_given_miscon):]
        selected_miscon = get_bestmiscon(best_response)

    return selected_miscon


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
