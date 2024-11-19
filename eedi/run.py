import logging
import logging.handlers
import sys
from argparse import ArgumentParser
from dataclasses import dataclass

from eedi import (
    MODEL_ID_ALL_MINILM_L6,
    MODEL_ID_ALL_MINILM_L6_FT,
    MODEL_ID_BGE_LARGE_EN,
    MODEL_ID_BGE_LARGE_EN_FT,
    MODEL_ID_LLAMA31_8B,
    MODEL_ID_LLAMA32_3B,
    MODEL_ID_MSMARCO_MINILM_L6,
    MODEL_ID_MSMARCO_MINILM_L6_FT,
    MODEL_ID_PARAPHRASE_MINILM_L6,
    MODEL_ID_PARAPHRASE_MINILM_L6_FT,
    MODEL_ID_QWEN25_7B,
    SEED,
)
from eedi.experiment import run_experiment
from eedi.utils import get_logger

LLMS = {
    "llama3b": MODEL_ID_LLAMA32_3B,
    "llama8b": MODEL_ID_LLAMA31_8B,
    "qwen7b": MODEL_ID_QWEN25_7B,
}
KNOWLEDGES = ["none", "genk", "tot"]
ENCODERS = {
    "bge": MODEL_ID_BGE_LARGE_EN,
    "bge-ft": MODEL_ID_BGE_LARGE_EN_FT,
    "allmini": MODEL_ID_ALL_MINILM_L6,
    "allmini-ft": MODEL_ID_ALL_MINILM_L6_FT,
    "marcomini": MODEL_ID_MSMARCO_MINILM_L6,
    "marcomini-ft": MODEL_ID_MSMARCO_MINILM_L6_FT,
    "paramini": MODEL_ID_PARAPHRASE_MINILM_L6,
    "paramini-ft": MODEL_ID_PARAPHRASE_MINILM_L6_FT,
}


@dataclass(frozen=True)
class RunConfig:
    llm: str = "llama3b"
    knowledge: str = "none"
    encoder: str = "bge"
    enable_tqdm: bool = False
    seed: int = SEED
    dataset: str = "test"
    sample_size: int = -1
    batch_size: int = 20


def get_run_config() -> RunConfig:
    parser = ArgumentParser()
    parser.add_argument("--enable-tqdm", action="store_true", help="enables tqdm")
    parser.add_argument(
        "-m",
        "--llm",
        type=str,
        choices=list(LLMS.keys()),
        default="llama3b",
        help="llm model name",
    )
    parser.add_argument(
        "-k",
        "--knowledge",
        type=str,
        choices=KNOWLEDGES,
        default="none",
        help="added knowledge context",
    )
    parser.add_argument(
        "-e",
        "--encoder",
        type=str,
        choices=list(ENCODERS.keys()),
        default="bge",
        help="sentence encoder",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=SEED,
        help="random seed",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="test",
        choices=["train", "test"],
        help="evaluate on train or test set",
    )
    parser.add_argument("-n", "--sample-size", type=int, default=-1, help="size of data to infer on, -1 to run all data")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=20,
        help="inference batch size",
    )
    args = parser.parse_args()
    return RunConfig(**vars(args))


def init_logger():
    logger = get_logger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)


def main() -> None:
    init_logger()
    run_conf = get_run_config()
    logger = get_logger()
    logger.info(f"run config : {vars(run_conf)}")
    run_experiment(
        llm_id=LLMS[run_conf.llm],
        knowledge=run_conf.knowledge,
        encoder_id=ENCODERS[run_conf.encoder],
        enable_tqdm=run_conf.enable_tqdm,
        seed=run_conf.seed,
        dataset=run_conf.dataset,
        sample_size=run_conf.sample_size,
        batch_size=run_conf.batch_size,
    )


if __name__ == "__main__":
    main()
