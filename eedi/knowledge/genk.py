from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from transformers.generation import GenerationMixin
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from functools import partial
from typing import Callable
from eedi.utils import get_device, save_df
import torch
from tqdm import tqdm
from torch import Tensor
from eedi import RESULTS_DIR

def enhance_with_knowledge(
    *,
    llm: GenerationMixin,
    llm_tokenizer: PreTrainedTokenizerFast,
    encoder: SentenceTransformer,
    df_xy: DataFrame,
    batch_size: int,
    disable_tqdm: bool,
    knowledge_template_func: Callable,
    remove_prompt_func: Callable,
    run_id: str,
) -> DataFrame:
    df_xy_enhanced = df_xy.copy(deep=True)
    df_xy_enhanced["KnowledgePrompt"] = df_xy_enhanced.apply(
        partial(knowledge_template_func, tokenizer=llm_tokenizer),
        axis=1,
    )
    knowledge_prompts = df_xy_enhanced["KnowledgePrompt"].to_list()
    knowledge_prompt_batches = [knowledge_prompts[i : i + batch_size] for i in range(0, len(knowledge_prompts), batch_size)]
    kp_model_inputs_batches = []
    for kpb in knowledge_prompt_batches:
        kp_model_inputs = llm_tokenizer(kpb, return_tensors="pt", padding=True).to(get_device())
        kp_model_inputs_batches.append(kp_model_inputs)
    llm.eval()
    with torch.no_grad():
        kp_output_ids_batches: list[Tensor] = []
        for tokens in tqdm(kp_model_inputs_batches, disable=disable_tqdm):
            output_ids_batch: Tensor = llm.generate(
                tokens.input_ids,
                max_new_tokens=4096,
                num_return_sequences=1,
                attention_mask=tokens.attention_mask,
            )
            kp_output_ids_batches.append(output_ids_batch.cpu())
    kp_responses = []
    for output_ids_batch in kp_output_ids_batches:
        kp_responses.extend(llm_tokenizer.batch_decode(output_ids_batch))
    resp_key = "KnowledgeFullResponse"
    df_xy_enhanced[resp_key] = kp_responses
    df_xy_enhanced["Knowledge"] = df_xy_enhanced.apply(partial(remove_prompt_func, resp_key=resp_key), axis=1)
    save_df(df_xy_enhanced, RESULTS_DIR, run_id, "df_xy_enhanced.parquet")
    return df_xy_enhanced
