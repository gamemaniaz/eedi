# from baseline.py
def apk(actual, predicted, k=25):
    if not actual:
        return 0.0

    actual = [actual]
    # comment below line if predicted is already a list
    predicted = list(map(int, predicted.split()))

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


# from baseline.py
def mapk(actual, predicted, k=25):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def process_pred(df):
    df[["QuestionId", "Answer"]] = df["QuestionId_Answer"].str.split("_", expand=True)

    target = []
    pred = []
    for _, row in df.iterrows():
        question_id = row["QuestionId"]
        answer = row["Answer"]
        if int(question_id) in data_df["QuestionId"].values:
            misconception_column = f"Misconception{answer}Id"
            r = data_df[data_df["QuestionId"] == int(question_id)]
            targ = int(r[misconception_column].values[0])
            target.append(targ)
            pred.append(row["MisconceptionId"])
    return mapk(target, pred)


df = pd.read_parquet("df_submission.parquet")
print(process_pred(df))
