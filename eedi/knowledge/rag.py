from typing import Callable

from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers.generation import GenerationMixin
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from eedi import RESULTS_DIR, TRAIN_SET_CSV
from eedi.preprocess import FilterOption, filter_data, get_miscon, preproc_base_data
from eedi.utils import save_df

prompt_template = """Question: {Question}
Incorrect Answer: {IncorrectAnswer}
Correct Answer: {CorrectAnswer}
Construct Name: {ConstructName}
Subject Name: {SubjectName}"""


NUM_KNOWLEDGE = 10


def generate_knowledge(
    *,
    rag_source: DataFrame,
    ConstructName: str,
) -> str:
    knowledge = rag_source[rag_source["ConstructName"] == ConstructName]["MisconceptionName"].to_list()[:25]
    if len(knowledge) < NUM_KNOWLEDGE:
        questions_list = rag_source["QuestionText"].to_list()
        vectorizer = TfidfVectorizer().fit_transform(questions_list)
        similarity_matrix = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()
        similar_indices = similarity_matrix.argsort()[::-1]
        similar_misconceptions = rag_source.iloc[similar_indices]["MisconceptionName"].tolist()
        knowledge.extend(similar_misconceptions)
    knowledge = knowledge[:NUM_KNOWLEDGE]
    knowledge_str = "\n".join([f"<possible-misconception>{m}</possible-misconception>" for m in knowledge])
    return knowledge_str


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
    """should return dataframe with new knowledge column"""
    df_xy_enhanced = df_xy.copy(deep=True)

    df_miscon = get_miscon()
    df_xy = preproc_base_data(
        df_miscon=df_miscon,
        dataset=TRAIN_SET_CSV,
        run_id=run_id,
        persist=False,
    )
    rag_source = filter_data(
        df_xy=df_xy,
        filter_option=FilterOption.XM,
        run_id=run_id,
        persist=False,
    )

    knowledges = []
    for _, row in tqdm(df_xy_enhanced.iterrows(), desc="gentot"):
        knowledges.append(
            generate_knowledge(
                rag_source=rag_source,
                ConstructName=row["ConstructName"],
            )
        )
    df_xy_enhanced["Knowledge"] = knowledges
    save_df(df_xy_enhanced, RESULTS_DIR, run_id, "df_xy_enhanced.parquet")
    return df_xy_enhanced
