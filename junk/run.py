import re
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from tqdm import tqdm

model_id = "meta-llama/Llama-3.2-3B-Instruct"
train_file = "data/train.csv"
test_file = "data/test.csv"


df_train = pd.read_csv(
    "data/train.csv",
    dtype={
        "MisconceptionAId": "Int64",
        "MisconceptionBId": "Int64",
        "MisconceptionCId": "Int64",
        "MisconceptionDId": "Int64",
    },
).fillna(-1)
df_test = pd.read_csv("data/test.csv").head(1)

PROMPT = """Question: {Question}
Incorrect Answer: {IncorrectAnswer}
Correct Answer: {CorrectAnswer}
Construct Name: {ConstructName}
Subject Name: {SubjectName}

Your main task is to explain the misconception behind Incorrect Answer. Before answering the next task, think step by step concisely in 1-2 sentences inside the tag <response>$$INSERT TEXT HERE$$</response>."""

def apply_template(row, tokenizer):
    messages = [
        {
            "role": "user",
            "content": PROMPT.format(
                ConstructName=row["ConstructName"],
                SubjectName=row["SubjectName"],
                Question=row["QuestionText"],
                IncorrectAnswer=row[f"CorrectAnswerText"],
                CorrectAnswer=row[f"AnswerText"],
            ),
        }
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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
    else:
        return None

df_test["CorrectAnswerText"] = df_test.apply(get_correct_answer, axis=1)
select_column = [
    "QuestionId",
    "ConstructName",
    "SubjectName",
    "CorrectAnswer",
    "QuestionText",
    "CorrectAnswerText",
]
df_answer = pd.melt(
    df_test,
    id_vars=select_column,
    value_vars=[f"Answer{ans}Text" for ans in ["A", "B", "C", "D"]],
    var_name="Option",
    value_name="AnswerText",
).sort_values("QuestionId")

def process_option(x):
    out = re.search(r"Answer([A-D])", x)
    if out:
        return out.group(1)
    return None

df_answer["Option"] = df_answer["Option"].apply(process_option)

tokenizer = AutoTokenizer.from_pretrained(model_id)
custom_chattemp = Path("chattemplate.j2").read_text(encoding="UTF-8")
tokenizer.chat_template = custom_chattemp
# tokenizer.chat_template = (
#     "{{bos_token}}"
#     "{% for message in messages %}"
#         "<|start_header_id|>{{message['role']}}<|end_header_id|>\n\n{{message['content']}}<|eot_id|>"
#     "{% endfor %}"
#     "{% if add_generation_prompt%}<|start_header_id|>teacher<|end_header_id|>\n\n"
#         "{% else %}{{eos_token}}"
#     "{% endif %}"
# )
df_answer = df_answer[df_answer["CorrectAnswer"] != df_answer["Option"]]
df_answer["Prompt"] = df_answer.apply(
    lambda row: apply_template(row, tokenizer), axis=1
)
df_answer.to_parquet("test.parquet", index=False)

df = pd.read_parquet("test.parquet")
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )
pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"temperature": 0.7, "top_p": 0.9},
    # device="auto",
)

responses = []
for v in tqdm(df["Prompt"].values):
    out = pipeline(v, max_new_tokens=5192)
    print(out)
    responses.append(out)

responses = [x[0]["generated_text"] for x in responses]
df["FullResponse"] = responses

def extract_response(text):
    return ",".join(re.findall(r"<response>(.*?)</response>", text)).strip()

responses = [extract_response(x) for x in responses]
df["Misconception"] = responses
df.to_parquet("output.parquet", index=False)

df = pd.read_parquet("output.parquet")

print("--------------------------")
print(df["FullResponse"][0])
print("--------------------------")
print(df["FullResponse"][1])
print("--------------------------")
print(df["FullResponse"][2])
print("--------------------------")


## generate misconception from llm
## compare generated misconception with misconception mapping descriptions to retrieve mapping