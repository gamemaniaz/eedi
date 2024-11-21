from pathlib import Path

# main paths
PROJECT_ROOT = Path(__file__).parents[1].resolve()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / ".res"

# data paths
ORIGINAL_TRAIN_CSV = DATA_DIR / "train.csv"
ORIGINAL_TEST_CSV = DATA_DIR / "test.csv"
TRAIN_SET_CSV = DATA_DIR / "train_data.csv"
TEST_SET_CSV = DATA_DIR / "test_data.csv"
MISCONCEPTIONS_CSV = DATA_DIR / "misconception_mapping.csv"

# constants
SEED = 20241101
MODEL_ID_LLAMA32_3B = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_ID_LLAMA31_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_ID_QWEN25_7B = "Qwen/Qwen2.5-7B-Instruct"
MODEL_ID_BGE_LARGE_EN = "BAAI/bge-large-en-v1.5"
MODEL_ID_BGE_LARGE_EN_FT = "/retrieve/from/some/path"  # TODO edit on server
MODEL_ID_ALL_MINILM_L6 = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_ID_ALL_MINILM_L6_FT = "/retrieve/from/some/path"  # TODO edit on server
MODEL_ID_MSMARCO_MINILM_L6 = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
MODEL_ID_MSMARCO_MINILM_L6_FT = "/retrieve/from/some/path"  # TODO edit on server
MODEL_ID_PARAPHRASE_MINILM_L6 = "sentence-transformers/paraphrase-MiniLM-L6-v2"
MODEL_ID_PARAPHRASE_MINILM_L6_FT = "/retrieve/from/some/path"  # TODO edit on server
KNOWLEDGE_TYPE_NONE = "none"
KNOWLEDGE_TYPE_GENK = "genk"
KNOWLEDGE_TYPE_TOT = "tot"
KNOWLEDGE_TYPE_RAG = "rag"

# maps
LLMS = {
    "llama3b": MODEL_ID_LLAMA32_3B,
    "llama8b": MODEL_ID_LLAMA31_8B,
    "qwen7b": MODEL_ID_QWEN25_7B,
}
KNOWLEDGES = [
    KNOWLEDGE_TYPE_NONE,
    KNOWLEDGE_TYPE_GENK,
    KNOWLEDGE_TYPE_TOT,
    KNOWLEDGE_TYPE_RAG,
]
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
