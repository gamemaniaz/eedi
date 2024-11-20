import logging
import logging.handlers
import sys
from argparse import ArgumentParser
from dataclasses import dataclass

from eedi import ENCODERS, KNOWLEDGES, LLMS, SEED
from eedi.experiment import run_experiment
from eedi.utils import get_logger


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
