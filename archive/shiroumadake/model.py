from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def prepare_llama_model(model=True, tokenizer=True) -> tuple[LlamaForCausalLM, PreTrainedTokenizerFast]:
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model = None
    tokenizer = None
    if model:
        model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_id)
    if tokenizer:
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer
