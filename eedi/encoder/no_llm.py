import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

processed_df = pd.read_csv("data/processed_df.csv")
processed_test_df = pd.read_csv("data/processed_test_df.csv")


def get_top_25_misconceptions(row, processed_df):
    same_subject = []
    if len(same_subject) < 25:
        same_subject.extend(processed_df[processed_df["ConstructName"] == row["ConstructName"]]["MisconceptionId"].tolist())

    if len(same_subject) < 25:
        all_texts = processed_df["QuestionText"].tolist() + [row["QuestionText"]]
        vectorizer = TfidfVectorizer().fit_transform(all_texts)
        similarity_matrix = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()

        similar_indices = similarity_matrix.argsort()[::-1]
        similar_misconceptions = processed_df.iloc[similar_indices]["MisconceptionId"].tolist()

        for mis_id in similar_misconceptions:
            if len(same_subject) >= 15:
                break
            if mis_id not in same_subject:
                same_subject.append(mis_id)

    if len(same_subject) < 25:
        same_construct = processed_df[processed_df["SubjectName"] == row["SubjectName"]]["MisconceptionId"].tolist()
        same_subject.extend([mis_id for mis_id in same_construct if mis_id not in same_subject])

    return same_subject[:25]


def generate_predictions(processed_test_df, processed_df):
    actual_values = processed_test_df["MisconceptionId"].tolist()
    predicted_values = []

    for _, row in processed_test_df.iterrows():
        predictions = get_top_25_misconceptions(row, processed_df)
        predicted_values.append(predictions)

    return actual_values, predicted_values


actual_values, predicted_values = generate_predictions(processed_test_df, processed_df)
