import re
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.tokenization_utils_base import BatchEncoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)

eedi_train_csv = "data/train.csv"
eedi_test_csv = "data/test.csv"
eedi_miscon_csv = "data/misconception_mapping.csv"
llm_model_id = "meta-llama/Llama-3.2-1B-Instruct"
sbert_model_id = "BAAI/bge-small-en-v1.5"
submission_csv = "submission.csv"
intermediate_dir = ".intm"
random_seed = 20241030
sample_size = -1

prompt = """Question: {Question}
Incorrect Answer: {IncorrectAnswer}
Correct Answer: {CorrectAnswer}
Construct Name: {ConstructName}
Subject Name: {SubjectName}

Your task: Identify the misconception behind Incorrect Answer. Answer concisely and generically inside <response>$$INSERT TEXT HERE$$</response>.
Before answering the question think step by step concisely in 1-2 sentence inside <thinking>$$INSERT TEXT HERE$$</thinking> tag and respond your final misconception inside <response>$$INSERT TEXT HERE$$</response> tag."""


def apply_template(row, tokenizer):
    messages = [
        {
            "role": "user",
            "content": prompt.format(
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


def remove_prompt(record):
    l = len(record["Prompt"])
    value = record["FullResponse"][l:]
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
    d = Path(intermediate_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / fn
    if p.exists():
        raise FileExistsError(p.as_posix())
    df.to_parquet(p, index=False)


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


def tokenize_for_llm(
    tokenizer: Any,
    df_xy: pd.DataFrame,
    *,
    persist: bool = False,
    persist_fn: str = "df_prompt.parquet",
    run_id: str = None,
) -> tuple[pd.DataFrame, BatchEncoding]:
    df_prompt = df_xy.copy(deep=True)
    df_prompt["Prompt"] = df_prompt.apply(
        partial(apply_template, tokenizer=tokenizer),
        axis=1,
    )
    df_prompt = df_prompt[["QuestionId_Answer", "Prompt"]]
    dfpersist(persist, df_prompt, intermediate_dir, run_id, persist_fn)
    model_inputs = tokenizer(df_prompt["Prompt"].to_list(), return_tensors="pt", padding=True).to(device)
    return df_prompt, model_inputs


def generate_zeroshot(
    model: LlamaForCausalLM,
    tokenizer,
    tokens,
    df_prompt,
    *,
    persist: bool = False,
    persist_fn: str = "df_responses.parquet",
    run_id: str = None,
) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(tokens.input_ids, max_new_tokens=4096, num_return_sequences=1)
    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)  # list[str]
    df_prompt["FullResponse"] = responses
    df_prompt["Misconception"] = [extract_response(x) for x in df_prompt["FullResponse"]]
    dfpersist(persist, df_prompt, intermediate_dir, run_id, persist_fn)
    return df_prompt


def generate_misconceptions(
    model: SentenceTransformer,
    df_responses,
    df_miscon,
    *,
    persist: bool = False,
    persist_fn: str = "df_submission.parquet",
    run_id: str = None,
):
    print(">> generate_misconceptions")
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
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=25):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def evaluate(df_xy, df_submission, *, run_id: str, fn: str = "results.txt"):
    results = mapk(df_xy["MisconceptionId"].to_list(), df_submission["MisconceptionId"].to_list())
    assert run_id is not None
    d = Path(intermediate_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)
    p: Path = d / fn
    if p.exists():
        raise FileExistsError(p.as_posix())
    p.write_text(f"Results of MapK=25 : {results}")
    print("Results of MapK=25 :", results)


def main() -> None:
    run_id = str(uuid4())
    print("run_id:", run_id)
    df_xy, df_miscon = prepare_base_data(persist=True, run_id=run_id)
    df_xy = filter_data(df_xy, persist=True, run_id=run_id)
    if sample_size > 0:
        df_xy = df_xy.sample(sample_size)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(llm_model_id).to(device)
    sbert_model = SentenceTransformer(sbert_model_id)
    df_prompt, tokens = tokenize_for_llm(tokenizer, df_xy, persist=True, run_id=run_id)
    df_responses = generate_zeroshot(model, tokenizer, tokens, df_prompt, persist=True, run_id=run_id)
    df_submission = generate_misconceptions(sbert_model, df_responses, df_miscon, persist=True, run_id=run_id)
    evaluate(df_xy, df_submission, run_id=run_id)


if __name__ == "__main__":
    main()
