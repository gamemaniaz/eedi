def process_pred(df):
    df[['QuestionId', 'Answer']] = df['QuestionId_Answer'].str.split('_', expand=True)

    target = []
    pred = []
    for _, row in df.iterrows():
        question_id = row['QuestionId']
        answer = row['Answer']
        if int(question_id) in data_df['QuestionId'].values:
            misconception_column = f'Misconception{answer}Id'
            r = data_df[data_df['QuestionId'] == int(question_id)]
            targ = int(r[misconception_column].values[0])
            target.append(targ)
            pred.append(row['MisconceptionId'])
    return mapk(target, pred)

df = pd.read_parquet('df_submission.parquet')
print(process_pred(df))