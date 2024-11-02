import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = 'dataset'
RETRIEVE_NUM = 25

MODEL_OUTPUT_PATH = 'model/'

test = pd.read_csv(f"{DATA_PATH}/test.csv")
misconception_mapping = pd.read_csv(f"{DATA_PATH}/misconception_mapping.csv")


common_col = [
    "QuestionId",
    "ConstructName",
    "SubjectName",
    "QuestionText",
    "CorrectAnswer",
]

test_long = (
    test[common_col + [f"Answer{alpha}Text" for alpha in ["A", "B", "C", "D"]]]
    .melt(
        id_vars=common_col,
        value_vars=[f"Answer{alpha}Text" for alpha in ["A", "B", "C", "D"]],
        var_name="AnswerType",
        value_name="AnswerText"
    )
)
test_long["AllText"] = test_long["ConstructName"] + " " + test_long["SubjectName"] + " " + test_long["QuestionText"] + " " + test_long["AnswerText"]
test_long["AnswerAlphabet"] = test_long["AnswerType"].str.extract(r"Answer([A-D])Text$")
test_long["QuestionId_Answer"] = test_long["QuestionId"].astype(str) + "_" + test_long["AnswerAlphabet"]

model = SentenceTransformer(MODEL_OUTPUT_PATH)

test_long_vec = model.encode(
    test_long["AllText"].to_list(), normalize_embeddings=True
)
misconception_mapping_vec = model.encode(
    misconception_mapping["MisconceptionName"].to_list(), normalize_embeddings=True
)
print(test_long_vec.shape)
print(misconception_mapping_vec.shape)

test_cos_sim_arr = cosine_similarity(test_long_vec, misconception_mapping_vec)
test_sorted_indices = np.argsort(-test_cos_sim_arr, axis=1)[:, :RETRIEVE_NUM]


test_long["MisconceptionId"] = [" ".join(map(str, indices)) for indices in test_sorted_indices]
test_long["MisconceptionText"] = ["\n".join(misconception_mapping.iloc[indices]["MisconceptionName"].values) for indices in test_sorted_indices]

# Filter where CorrectAnswer != AnswerAlphabet
filtered_test_long = test_long[test_long["CorrectAnswer"] != test_long["AnswerAlphabet"]]

# Select relevant columns and sort by QuestionId_Answer
submission = filtered_test_long[["QuestionId_Answer", "MisconceptionId"]].sort_values(by="QuestionId_Answer")
submission.to_csv("submission.csv", index=False)
