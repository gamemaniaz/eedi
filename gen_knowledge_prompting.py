# %%
import re
import gc
import shutil
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)

# %%
eedi_train_csv = "data/train.csv"
eedi_test_csv = "data/test.csv"
eedi_miscon_csv = "data/misconception_mapping.csv"
llm_model_id = "meta-llama/Llama-3.2-3B-Instruct"
sbert_model_id = "/home/e/e1374073/models/bge-large-en-finetune-v1"
# sbert_model_id = ".tmp/bge-large-en-finetune-v1"
submission_csv = "submission.csv"
intermediate_dir = ".intm"
last_dir = ".last"
random_seed = 20241030
sample_size = -1  # -1 to run all data
batch_size = 20
disable_tqdm = True

# %%
knowledge_prompt = """
Task Construct Name : {ConstructName}
Subject Name : {SubjectName}

Given the mathematical subject and task construct, explain how a student can perform such a task step by step, in at most 5 steps.
"""
knowledge_prompt = knowledge_prompt.strip()

# %%
task_prompt = """
<knowledge>
{Knowledge}
</knowledge>

<task-context>
Question: {Question}
Incorrect Answer: {IncorrectAnswer}
Correct Answer: {CorrectAnswer}
Construct Name: {ConstructName}
Subject Name: {SubjectName}
</task-context>

Your task: Identify the misconception behind Incorrect Answer. Answer concisely and generically inside <response>$$INSERT TEXT HERE$$</response>. Before answering the math question, think step by step concisely in 1-2 sentence inside <thinking>$$INSERT TEXT HERE$$</thinking> tag and respond your final misconception inside <response>$$INSERT TEXT HERE$$</response> tag.
"""
task_prompt = task_prompt.strip()

# %%
def apply_knowledge_template(row, tokenizer):
    messages = [
        {
            "role": "user",
            "content": knowledge_prompt.format(
                ConstructName=row["ConstructName"],
                SubjectName=row["SubjectName"],
            ),
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def apply_task_template(row, tokenizer):
    messages = [
        {
            "role": "user",
            "content": task_prompt.format(
                Knowledge=row["Knowledge"],
                ConstructName=row["ConstructName"],
                SubjectName=row["SubjectName"],
                Question=row["QuestionText"],
                IncorrectAnswer=row[f"CorrectAnswerText"],
                CorrectAnswer=row[f"AnswerText"],
            ),
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def get_correct_answer(row):
    if row["CorrectAnswer"] == "A":
        return row["AnswerAText"]
    elif row["CorrectAnswer"] == "B":
        return row["AnswerBText"]
    elif row["CorrectAnswer"] == "C":
        return row["AnswerCText"]
    elif row["CorrectAnswer"] == "D":
        return row["AnswerDText"]
    return None


def process_option(x, regex):
    result = re.search(regex, x)
    return str(result.group(1)) if result else ""


def remove_prompt(record, resp_key: str):
    l = record[resp_key].index("<|start_header_id|>assistant<|end_header_id|>") + 45
    value = record[resp_key][l:].strip()
    if value == "":
        value = "No Misconception Found"
    return value


def extract_response(text):
    subresponses = re.findall(r"<response>(?s:.*?)</response>", text)
    subresponses = [x.strip().replace("<response>", "").replace("</response>", "") for x in subresponses]
    return " ".join(subresponses).strip()


def dfpeek(title: str, df: pd.DataFrame) -> None:
    print(">>>>>>>>>>", title, ">>>>>>>>>")
    display(df.head(1).transpose())
    print("<<<<<<<<<<", title, "<<<<<<<<<<", end="\n\n")


def dfpersist(trigger: bool, df: pd.DataFrame, int_dir: str, run_id: str, fn: str) -> None:
    if not trigger:
        return
    assert run_id is not None
    d = Path(int_dir) / run_id
    d_last = Path(int_dir) / last_dir
    d.mkdir(parents=True, exist_ok=True)
    p = d / fn
    if p.exists():
        raise FileExistsError(p.as_posix())
    d_last.mkdir(parents=True, exist_ok=True)
    p_last = d_last / fn
    df.to_parquet(p, index=False)
    shutil.copyfile(p, p_last)

# %%
def prepare_base_data(*, persist: bool = False, run_id: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    # read info
    df = pd.read_csv(
        eedi_train_csv,
        dtype={
            "MisconceptionAId": "Int64",
            "MisconceptionBId": "Int64",
            "MisconceptionCId": "Int64",
            "MisconceptionDId": "Int64",
        },
    ).fillna(-1)
    df_miscon = pd.read_csv(
        eedi_miscon_csv,
        dtype={
            "MisconceptionId": "Int64",
        },
    )

    # store correct answer
    df["CorrectAnswerText"] = df.apply(get_correct_answer, axis=1)

    # pivot out each wrong answer into its own row, currently the 3 wrong answers are within the same record
    df_x = df.melt(
        id_vars=[
            "QuestionId",
            "ConstructName",
            "SubjectName",
            "QuestionText",
            "CorrectAnswer",
            "CorrectAnswerText",
        ],
        value_vars=[
            "AnswerAText",
            "AnswerBText",
            "AnswerCText",
            "AnswerDText",
        ],
        var_name="Option",
        value_name="AnswerText",
    )
    df_y = df.melt(
        id_vars=[
            "QuestionId",
        ],
        value_vars=[
            "MisconceptionAId",
            "MisconceptionBId",
            "MisconceptionCId",
            "MisconceptionDId",
        ],
        var_name="Option",
        value_name="MisconceptionId",
    )

    # remap option values of from "xxxxXxxxx" to "X"
    df_x["Option"] = df_x["Option"].map(partial(process_option, regex=r"Answer([A-D])Text"))
    df_y["Option"] = df_y["Option"].map(partial(process_option, regex=r"Misconception([A-D])Id"))

    # mark correct answers
    df_x["IsCorrectAnswer"] = df_x["CorrectAnswer"] == df_x["Option"]

    # create primary key, drop components, reorder col
    df_x["QuestionId_Answer"] = df_x["QuestionId"].astype(str) + "_" + df_x["Option"].astype(str)
    df_y["QuestionId_Answer"] = df_y["QuestionId"].astype(str) + "_" + df_y["Option"].astype(str)
    df_x.drop(columns=["QuestionId", "Option"], inplace=True)
    df_y.drop(columns=["QuestionId", "Option"], inplace=True)
    df_x = df_x[["QuestionId_Answer"] + [c for c in df_x.columns if c != "QuestionId_Answer"]]
    df_y = df_y[["QuestionId_Answer"] + [c for c in df_y.columns if c != "QuestionId_Answer"]]

    # map misconception text to labels
    df_y = df_y.join(df_miscon, on="MisconceptionId", how="left", lsuffix="a", rsuffix="b")
    df_y = df_y[["QuestionId_Answer", "MisconceptionId", "MisconceptionName"]]

    # merge datasets
    df_xy = df_x.merge(df_y, how="left", on="QuestionId_Answer")

    # persist df_xy
    dfpersist(persist, df_xy, intermediate_dir, run_id, "df_xy.parquet")

    return df_xy, df_miscon

# %%
def filter_by_wrong_answers_only(df_xy: pd.DataFrame) -> pd.DataFrame:
    df_xy = df_xy[~df_xy["IsCorrectAnswer"]]
    return df_xy


def filter_by_wrong_answers_and_misconceptions(df_xy: pd.DataFrame) -> pd.DataFrame:
    return df_xy[(~df_xy["IsCorrectAnswer"]) & (df_xy["MisconceptionId"] != -1)]


def no_filter(df_xy: pd.DataFrame) -> pd.DataFrame:
    return df_xy


def filter_data(df_xy: pd.DataFrame, persist: bool = False, run_id: str = None) -> pd.DataFrame:
    filter_func = filter_by_wrong_answers_and_misconceptions
    df_xy_filtered = filter_func(df_xy)
    dfpersist(persist, df_xy_filtered, intermediate_dir, run_id, "df_xy_filtered.parquet")
    return df_xy_filtered

# %%
run_id = str(uuid4())
print("run_id:", run_id)
df_xy, df_miscon = prepare_base_data(persist=True, run_id=run_id)
df_xy = filter_data(df_xy, persist=True, run_id=run_id)
if sample_size > 0:
    df_xy = df_xy.sample(sample_size)
tokenizer = AutoTokenizer.from_pretrained(llm_model_id, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(llm_model_id).to(device)
model.generation_config.pad_token_id = tokenizer.pad_token_id
sbert_model = SentenceTransformer(sbert_model_id)

# %%
def generate_knowledge(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    df_xy: pd.DataFrame,
    *,
    persist: bool = False,
    persist_fn: str = "df_xy_enhanced.parquet",
    run_id: str = None,
):
    print(">> generating knowledge")
    df_xy_enhanced = df_xy.copy(deep=True)
    df_xy_enhanced["KnowledgePrompt"] = df_xy_enhanced.apply(
        partial(apply_knowledge_template, tokenizer=tokenizer),
        axis=1,
    )
    knowledge_prompts = df_xy_enhanced["KnowledgePrompt"].to_list()
    knowledge_prompt_batches = [knowledge_prompts[i : i + batch_size] for i in range(0, len(knowledge_prompts), batch_size)]
    kp_model_inputs_batches = []
    for kpb in knowledge_prompt_batches:
        kp_model_inputs = tokenizer(kpb, return_tensors="pt", padding=True).to(device)
        kp_model_inputs_batches.append(kp_model_inputs)
    model.eval()
    with torch.no_grad():
        kp_output_ids_batches: list[Tensor] = []
        for tokens in tqdm(kp_model_inputs_batches, disable=disable_tqdm):
            output_ids_batch: Tensor = model.generate(
                tokens.input_ids,
                max_new_tokens=4096,
                num_return_sequences=1,
                attention_mask=tokens.attention_mask,
            )
            kp_output_ids_batches.append(output_ids_batch.cpu())
    kp_responses = []
    for output_ids_batch in kp_output_ids_batches:
        kp_responses.extend(tokenizer.batch_decode(output_ids_batch))
    resp_key = "KnowledgeFullResponse"
    df_xy_enhanced[resp_key] = kp_responses
    df_xy_enhanced["Knowledge"] = df_xy_enhanced.apply(partial(remove_prompt, resp_key=resp_key), axis=1)
    dfpersist(persist, df_xy_enhanced, intermediate_dir, run_id, persist_fn)
    return df_xy_enhanced

# %%
def tokenize_for_llm(
    tokenizer: PreTrainedTokenizerFast,
    df_xy: pd.DataFrame,
    *,
    persist: bool = False,
    persist_fn: str = "df_prompt.parquet",
    run_id: str = None,
) -> tuple[pd.DataFrame, list[BatchEncoding]]:
    print(">> tokenizing for llm")
    df_prompt = df_xy.copy(deep=True)
    df_prompt["Prompt"] = df_prompt.apply(
        partial(apply_task_template, tokenizer=tokenizer),
        axis=1,
    )
    df_prompt = df_prompt[["QuestionId_Answer", "Prompt"]]
    dfpersist(persist, df_prompt, intermediate_dir, run_id, persist_fn)
    prompts = df_prompt["Prompt"].to_list()
    prompt_batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    model_inputs_batches = []
    for pb in prompt_batches:
        model_inputs = tokenizer(pb, return_tensors="pt", padding=True).to(device)
        model_inputs_batches.append(model_inputs)
    return df_prompt, model_inputs_batches

# %%
def generate_zeroshot(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    token_batches: list[BatchEncoding],
    df_prompt,
    *,
    persist: bool = False,
    persist_fn: str = "df_responses.parquet",
    run_id: str = None,
) -> pd.DataFrame:
    print(">> generating zeroshot misconceptions")
    model.eval()
    with torch.no_grad():
        output_ids_batches: list[Tensor] = []
        for tokens in tqdm(token_batches, disable=disable_tqdm):
            output_ids_batch: Tensor = model.generate(
                tokens.input_ids,
                max_new_tokens=4096,
                num_return_sequences=1,
                attention_mask=tokens.attention_mask,
            )
            output_ids_batches.append(output_ids_batch.cpu())
    responses = []
    for output_ids_batch in output_ids_batches:
        responses.extend(tokenizer.batch_decode(output_ids_batch))
    resp_key = "TaskResponse"
    df_prompt[resp_key] = responses
    df_prompt["Response"] = df_prompt.apply(partial(remove_prompt, resp_key=resp_key), axis=1)
    df_prompt["Misconception"] = [extract_response(x) for x in df_prompt["Response"]]
    dfpersist(persist, df_prompt, intermediate_dir, run_id, persist_fn)
    return df_prompt

# %%
def generate_misconceptions(
    model: SentenceTransformer,
    df_responses: pd.DataFrame,
    df_miscon: pd.DataFrame,
    *,
    persist: bool = False,
    persist_fn: str = "df_submission.parquet",
    run_id: str = None,
):
    print(">> finding similiar misconceptions")
    model.eval()
    with torch.no_grad():
        embedding_query = model.encode(df_responses["Misconception"].values)
        embedding_miscon = model.encode(df_miscon["MisconceptionName"].values)
    cosine_similarities = cosine_similarity(embedding_query, embedding_miscon)
    rev_sorted_indices = np.argsort(-cosine_similarities, axis=1)
    df_responses["MisconceptionId"] = rev_sorted_indices[:, :25].tolist()
    df_responses["MisconceptionId"] = df_responses["MisconceptionId"].apply(lambda x: " ".join(map(str, x)))
    df_submission = df_responses[["QuestionId_Answer", "MisconceptionId"]]
    dfpersist(persist, df_submission, intermediate_dir, run_id, persist_fn)
    return df_submission

# %%
def apk(actual, predicted, k=25):
    if not actual:
        return 0.0

    actual = [actual]
    predicted = list(map(int, predicted.split()))

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=25):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# %%
def evaluate(df_xy, df_submission, *, run_id: str, fn: str = "results.txt"):
    print(">> evaluating")
    results = mapk(df_xy["MisconceptionId"].to_list(), df_submission["MisconceptionId"].to_list())
    assert run_id is not None
    d = Path(intermediate_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)
    p: Path = d / fn
    if p.exists():
        raise FileExistsError(p.as_posix())
    p.write_text(f"Results of MapK=25 : {results}")
    print("Results of MapK=25 :", results)

# %%
def main() -> None:
    run_id = str(uuid4())
    print("run_id:", run_id)
    df_xy, df_miscon = prepare_base_data(persist=True, run_id=run_id)
    df_xy = filter_data(df_xy, persist=True, run_id=run_id)
    if sample_size > 0:
        df_xy = df_xy.sample(sample_size)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(llm_model_id).to(device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    df_xy_enhanced = generate_knowledge(model, tokenizer, df_xy, persist=True, run_id=run_id)
    df_prompt, token_batches = tokenize_for_llm(tokenizer, df_xy_enhanced, persist=True, run_id=run_id)
    df_responses = generate_zeroshot(model, tokenizer, token_batches, df_prompt, persist=True, run_id=run_id)

    del df_xy
    del tokenizer
    del model
    del df_prompt
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()

    sbert_model = SentenceTransformer(sbert_model_id)
    df_submission = generate_misconceptions(sbert_model, df_responses, df_miscon, persist=True, run_id=run_id)
    evaluate(df_xy_enhanced, df_submission, run_id=run_id)

# %%
main()


