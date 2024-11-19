from enum import Enum
from functools import partial
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from eedi import MISCONCEPTIONS_CSV, RESULTS_DIR
from eedi.utils import get_correct_answer, get_option, save_df


class FilterOption(Enum):
    X = 1
    XM = 2
    NOOP = 3


class FilterDataException(Exception):
    """Unexpected Filter Option"""


def get_miscon() -> DataFrame:
    return pd.read_csv(
        MISCONCEPTIONS_CSV,
        dtype={
            "MisconceptionId": "Int64",
        },
    )


def preproc_base_data(
    *,
    df_miscon: DataFrame,
    dataset: Path,
    run_id: str,
) -> DataFrame:
    # read info from train or test set
    df = pd.read_csv(
        dataset,
        dtype={
            "MisconceptionAId": "Int64",
            "MisconceptionBId": "Int64",
            "MisconceptionCId": "Int64",
            "MisconceptionDId": "Int64",
        },
    ).fillna(-1)

    # store correct answer
    df["CorrectAnswerText"] = df.apply(get_correct_answer, axis=1)

    # pivot out each wrong answer into its own row
    # currently the 3 wrong answers are within the same record
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
    df_x["Option"] = df_x["Option"].map(partial(get_option, regex=r"Answer([A-D])Text"))
    df_y["Option"] = df_y["Option"].map(partial(get_option, regex=r"Misconception([A-D])Id"))

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
    save_df(df_xy, RESULTS_DIR, run_id, "df_xy.parquet")

    return df_xy


def filter_by_wrong_answers_only(df_xy: DataFrame) -> DataFrame:
    return df_xy[~df_xy["IsCorrectAnswer"]]


def filter_by_wrong_answers_and_misconceptions(df_xy: DataFrame) -> DataFrame:
    return df_xy[(~df_xy["IsCorrectAnswer"]) & (df_xy["MisconceptionId"] != -1)]


def no_filter(df_xy: DataFrame) -> DataFrame:
    return df_xy


def filter_data(
    *,
    df_xy: pd.DataFrame,
    filter_option: FilterOption,
    run_id: str,
) -> pd.DataFrame:
    match filter_option:
        case FilterOption.X:
            filter_func = filter_by_wrong_answers_only
        case FilterOption.XM:
            filter_func = filter_by_wrong_answers_and_misconceptions
        case FilterOption.NOOP:
            filter_func = no_filter
        case _:
            raise FilterDataException(filter_option)
    df_xy_filtered = filter_func(df_xy)
    save_df(df_xy_filtered, RESULTS_DIR, run_id, "df_xy_filtered.parquet")
    return df_xy_filtered
