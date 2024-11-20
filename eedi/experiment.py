import gc
from functools import partial
from typing import Callable
from uuid import uuid4

import numpy as np
import torch
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationMixin
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from eedi import RESULTS_DIR, TEST_SET_CSV, TRAIN_SET_CSV
from eedi.eval import mapk
from eedi.knowledge import KNOWLEDGE_ENHANCER_MAP
from eedi.preprocess import FilterOption, filter_data, get_miscon, preproc_base_data
from eedi.prompt_processor import PROMPT_REMOVER_MAP, TEMPLATE_FUNC_MAP
from eedi.utils import get_device, get_logger, get_response, save_df


def build_task_prompts(
    *,
    tokenizer: PreTrainedTokenizerFast,
    df_xy: DataFrame,
    run_id: str,
    template_func: Callable,
    batch_size: int,
) -> tuple[DataFrame, list[BatchEncoding]]:
    df_prompt = df_xy.copy(deep=True)
    df_prompt["Prompt"] = df_prompt.apply(
        partial(template_func, tokenizer=tokenizer),
        axis=1,
    )
    df_prompt = df_prompt[["QuestionId_Answer", "Prompt"]]
    save_df(df_prompt, RESULTS_DIR, run_id, "df_prompt.parquet")
    prompts = df_prompt["Prompt"].to_list()
    prompt_batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    model_inputs_batches = []
    for pb in prompt_batches:
        model_inputs = tokenizer(pb, return_tensors="pt", padding=True).to(get_device())
        model_inputs_batches.append(model_inputs)
    return df_prompt, model_inputs_batches


def generate_responses(
    *,
    model: GenerationMixin,
    tokenizer: PreTrainedTokenizerFast,
    token_batches: list[BatchEncoding],
    df_prompt: DataFrame,
    remove_prompt_func: Callable,
    run_id: str,
    disable_tqdm: bool,
) -> DataFrame:
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
    df_responses = df_prompt.copy(deep=True)
    resp_key = "FullResponse"
    df_responses[resp_key] = responses
    df_responses["Response"] = df_responses.apply(
        f=partial(remove_prompt_func, resp_key=resp_key),
        axis=1,
    )
    df_responses["Misconception"] = [get_response(x) for x in df_responses["Response"]]
    save_df(df_responses, RESULTS_DIR, run_id, "df_responses.parquet")
    return df_responses


def generate_misconceptions(
    *,
    model: SentenceTransformer,
    df_responses: DataFrame,
    df_miscon: DataFrame,
    run_id: str = None,
):
    model.eval()
    with torch.no_grad():
        embedding_query = model.encode(df_responses["Misconception"].values)
        embedding_miscon = model.encode(df_miscon["MisconceptionName"].values)
    cosine_similarities = cosine_similarity(embedding_query, embedding_miscon)
    rev_sorted_indices = np.argsort(-cosine_similarities, axis=1)
    df_responses["MisconceptionId"] = rev_sorted_indices[:, :25].tolist()
    df_responses["MisconceptionId"] = df_responses["MisconceptionId"].apply(lambda x: " ".join(map(str, x)))
    df_submission = df_responses[["QuestionId_Answer", "MisconceptionId"]]
    save_df(df_submission, RESULTS_DIR, run_id, "df_submission.parquet")
    return df_submission


def evaluate(df_xy, df_submission, *, run_id: str, fn: str = "results.txt") -> None:
    results = mapk(
        actual=df_xy["MisconceptionId"].to_list(),
        predicted=df_submission["MisconceptionId"].to_list(),
        k=25,
    )
    p = RESULTS_DIR / run_id / fn
    p.write_text(f"Results of MapK=25 : {results}")
    get_logger().info(f"results of mapk=25 : {results}")


def run_experiment(
    *,
    llm_id: str,
    knowledge: str,
    encoder_id: str,
    enable_tqdm: bool,
    seed: int,
    dataset: str,
    sample_size: int,
    batch_size: int,
) -> None:
    logger = get_logger()

    run_id = str(uuid4())
    logger.info(f"run id : {run_id}")

    # TODO SEED ?

    disable_tqdm = not enable_tqdm
    dataset_fpath = TRAIN_SET_CSV if dataset == "train" else TEST_SET_CSV

    # prepare data
    df_miscon = get_miscon()
    df_xy = preproc_base_data(
        df_miscon=df_miscon,
        dataset=dataset_fpath,
        run_id=run_id,
    )
    df_xy_filtered = filter_data(
        df_xy=df_xy,
        filter_option=FilterOption.XM,
        run_id=run_id,
    )

    if sample_size > 0:
        df_xy_filtered = df_xy_filtered.sample(sample_size)

    # prepare model
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_id, padding_side="left")
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm: GenerationMixin = AutoModelForCausalLM.from_pretrained(
        llm_id,
        torch_dtype="auto",
        device_map="auto",
    )
    llm.generation_config.pad_token_id = llm_tokenizer.pad_token_id
    encoder = SentenceTransformer(encoder_id)

    # enhance with knowledge
    df_xy_enhanced = KNOWLEDGE_ENHANCER_MAP[knowledge](
        llm=llm,
        llm_tokenizer=llm_tokenizer,
        encoder=encoder,
        df_xy=df_xy_filtered,
        batch_size=batch_size,
        disable_tqdm=disable_tqdm,
    )

    # generate misconceptions
    df_prompt, token_batches = build_task_prompts(
        tokenizer=llm_tokenizer,
        df_xy=df_xy_enhanced,
        run_id=run_id,
        template_func=TEMPLATE_FUNC_MAP[llm_id][knowledge],
        batch_size=batch_size,
    )
    df_responses = generate_responses(
        model=llm,
        tokenizer=llm_tokenizer,
        token_batches=token_batches,
        df_prompt=df_prompt,
        remove_prompt_func=PROMPT_REMOVER_MAP[llm_id],
        run_id=run_id,
        disable_tqdm=disable_tqdm,
    )

    # cleanup unused
    del df_xy
    del df_xy_filtered
    del df_prompt
    del token_batches
    del llm
    del llm_tokenizer
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()

    # get similar misconceptions and evaluate
    df_submission = generate_misconceptions(
        model=encoder,
        df_responses=df_responses,
        df_miscon=df_miscon,
        run_id=run_id,
    )
    evaluate(df_xy_enhanced, df_submission, run_id=run_id)
