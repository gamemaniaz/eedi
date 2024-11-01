import os
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, pipeline

# env
is_kaggle = any(e for e in os.environ if "KAGGLE" in e.upper())
print("IS_KAGGLE:", is_kaggle)
if is_kaggle:
    print("KAGGLE_IS_COMPETITION_RERUN:", os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)

# constants
if is_kaggle:
    eedi_train_csv = "/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv"
    eedi_test_csv = "/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv"
    eedi_miscon_csv = "/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv"
    llm_model_id = "/kaggle/input/llama-3.2/transformers/3b-instruct/1"
    sbert_model_id = "/kaggle/input/bge-small-en-v1.5/transformers/bge/2"
else:
    eedi_train_csv = "data/train.csv"
    eedi_test_csv = "data/test.csv"
    eedi_miscon_csv = "data/misconception_mapping.csv"
    llm_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    sbert_model_id = "BAAI/bge-small-en-v1.5"
submission_csv = "submission.csv"
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


def process_option(x):
    result = re.search(r"Answer([A-D])", x)
    return str(result.group(1)) if result else ""


def prepare_df():
    df_test = pd.read_csv(eedi_test_csv)
    df_test["CorrectAnswerText"] = df_test.apply(get_correct_answer, axis=1)
    id_vars = [
        "QuestionId",
        "ConstructName",
        "SubjectName",
        "CorrectAnswer",
        "QuestionText",
        "CorrectAnswerText",
    ]
    value_vars = [f"Answer{ans}Text" for ans in ["A", "B", "C", "D"]]
    df = pd.melt(
        df_test,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="Option",
        value_name="AnswerText",
    ).sort_values("QuestionId")
    df["Option"] = df["Option"].map(process_option)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    df = df[df["CorrectAnswer"] != df["Option"]]
    df["Prompt"] = df.apply(lambda row: apply_template(row, tokenizer), axis=1)
    return df


def remove_prompt(record):
    l = len(record["Prompt"])
    value = record["FullResponse"][l:]
    return value


def extract_response(text):
    subresponses = re.findall(r"<response>(?s:.*?)</response>", text)
    subresponses = [x.strip().replace("<response>", "").replace("</response>", "") for x in subresponses]
    return " ".join(subresponses).strip()


def generate_zeroshot(df):
    generator = pipeline(
        "text-generation",
        model=llm_model_id,
        model_kwargs={"temperature": 0.7, "top_p": 0.9},
        device=device,
    )
    responses = []
    for v in df["Prompt"].values:
        out = generator(v, max_new_tokens=4096)
        responses.append(out)
    responses = [x[0]["generated_text"] for x in responses]
    df["FullResponse"] = responses
    df["FullResponse"] = df.apply(remove_prompt, axis=1)
    misconceptions = [extract_response(x) for x in df["FullResponse"]]
    df["Misconception"] = misconceptions
    return df


def generate_submission(df):
    df_miscon = pd.read_csv(eedi_miscon_csv)
    model = SentenceTransformer(sbert_model_id)
    embedding_query = model.encode(df["Misconception"])
    embedding_miscon = model.encode(df_miscon.MisconceptionName.values)
    cosine_similarities = cosine_similarity(embedding_query, embedding_miscon)
    rev_sorted_indices = np.argsort(-cosine_similarities, axis=1)
    df["MisconceptionId"] = rev_sorted_indices[:, :25].tolist()
    df["MisconceptionId"] = df["MisconceptionId"].apply(lambda x: " ".join(map(str, x)))
    df["QuestionId_Answer"] = df["QuestionId"].astype(str) + "_" + df["CorrectAnswer"]
    df_submission = df[["QuestionId_Answer", "MisconceptionId"]]
    df_submission.to_csv(submission_csv, index=False)
    return df_submission


def main():
    df = prepare_df()
    df = generate_zeroshot(df)
    df_submission = generate_submission(df)
    print(df_submission.head())


if __name__ == "__main__":
    main()
