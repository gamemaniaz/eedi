from configparser import ConfigParser
import pandas as pd


def ingest_data():
    config = ConfigParser()
    config.read("config.ini")
    section = config["DEFAULT"]

    df_train = pd.read_csv(section["train_csv"])
    df_test = pd.read_csv(section["test_csv"])
    df_sample_submission = pd.read_csv(section["sample_submission_csv"])
    df_misconception_mapping = pd.read_csv(section["misconception_mapping_csv"])

    return (
        df_train,
        df_test,
        df_sample_submission,
        df_misconception_mapping,
    )
