import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def assert_non_empty_str(v: str) -> None:
    assert isinstance(v, str), f"value {v} is not string"
    assert v and v.strip() != "", "save directory should not be empty"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def apply_template(prompt: str, row: pd.Series, tokenizer: Any) -> str:
    messages = [{"role": "user", "content": prompt.format(**row.to_dict())}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def get_correct_answer(row: pd.Series) -> str:
    option = row["CorrectAnswer"]
    if option not in {"A", "B", "C", "D"}:
        return None
    return row[f"Answer{option}Text"]


def get_option(x: str, regex: str) -> str:
    result = re.search(regex, x)
    if not result:
        return ""
    return str(result.group(1))


def get_response(text: str) -> str:
    subresponses = re.findall(r"<response>(?s:.*?)</response>", text)
    subresponses = [x.strip().replace("<response>", "").replace("</response>", "") for x in subresponses]
    return " ".join(subresponses).strip()


def save_df(df: pd.DataFrame, save_dirpath: Path, sub_dir: str, filename: str) -> None:
    assert_non_empty_str(sub_dir)
    d = save_dirpath / sub_dir
    d.mkdir(parents=True, exist_ok=True)
    p = d / filename
    assert not p.exists(), "df filepath already exists"
    df.to_parquet(p, index=False)


def get_logger():
    return logging.getLogger("experiment")
