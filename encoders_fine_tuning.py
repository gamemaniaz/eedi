#Use bi-encoder to find the top 25 misconceptions given question, and use cross-encoder to rerank

#read data
import pandas as pd
import numpy as np
miscon_df = pd.read_csv('misconception_mapping.csv')
train_df = pd.read_csv('train_data.csv')
train_df = train_df.sort_values(by='QuestionId')
test_df = pd.read_csv('test_data.csv')
test_df = test_df.sort_values(by='QuestionId')

#process data to obtain question-answer pairs
def process_df(data_df):
  df = pd.DataFrame()

  misconception_map = pd.Series(miscon_df.MisconceptionName.values, index=miscon_df.MisconceptionId).to_dict()
  rows = []
  for _, row in data_df.iterrows():
      incorrect_answers = {
          'A': (row['AnswerAText'], row['MisconceptionAId']),
          'B': (row['AnswerBText'], row['MisconceptionBId']),
          'C': (row['AnswerCText'], row['MisconceptionCId']),
          'D': (row['AnswerDText'], row['MisconceptionDId'])
      }
      for answer_key in ['A', 'B', 'C', 'D']:
          if answer_key == row['CorrectAnswer']:
              continue

          answer_text, misconception_id = incorrect_answers[answer_key]

          misconception_name = misconception_map.get(misconception_id, "Unknown")

          if not misconception_name == "Unknown":
            rows.append({
                'QuestionId': row['QuestionId'],
                'SubjectName': row['SubjectName'],
                'ConstructName': row['ConstructName'],
                'QuestionText': row['QuestionText'],
                'AnswerText': answer_text,
                'MisconceptionId': misconception_id,
                'MisconceptionName': misconception_name
            })

  df = pd.DataFrame(rows)

  return df

train_df = process_df(train_df)
test_df = process_df(test_df)

#Bi-Encoders
#sentence-transformers/all-MiniLM-L6-v2 - lightweight model
#sentence-transformers/msmarco-MiniLM-L6-cos-v5 - optimized for semantic search and retrieval, understanding context
#sentence-transformers/paraphrase-MiniLM-L6-v2 - fine-tuned on paraphrase data, effective at identifying similar meanings

#for bi-encoders, embeddings are done separately for query and candidate, so create embeddings once for all predictions
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_miscon_embeddings(model_name):
    #model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    misconceptions = miscon_df['MisconceptionName'].tolist()

    #encode misconceptions
    with torch.no_grad():
        misconception_embeddings = []
        for misconception in misconceptions:
            inputs = tokenizer(misconception, return_tensors="pt", padding=True)
            embedding = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
            misconception_embeddings.append(embedding.squeeze().numpy())

    return model, np.array(misconception_embeddings)


def bi_encoder_top_25_miscons(row, model, misconception_embeddings, tokenizer):
    query = row['SubjectName'] + '. ' + row['ConstructName'] + '. The question is ' + row['QuestionText'] + ' The student thinks the answer is ' + row['AnswerText']
    with torch.no_grad():
        inputs = tokenizer(query, return_tensors="pt", padding=True)
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()

    #calculate similarities and retrieve top-k misconceptions
    similarities = cosine_similarity([query_embedding], misconception_embeddings).flatten()
    top_k_indices = np.argsort(similarities)[-25:][::-1]

    return top_k_indices

#evaluation
def apk(actual, predicted, k=25):
    if not actual:
        return 0.0

    actual = [actual]

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    print(score / min(len(actual), k))
    return score / min(len(actual), k)

def mapk(actual, predicted, k=25):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def evaluate_bi_encoder(model_name):
  model, miscon_embeddings = get_miscon_embeddings(model_name)
  contains_count = 0
  mini_l6_v2_preds = []
  actual = []
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  for i, row in test_df.iterrows():
    pred_list = bi_encoder_top_25_miscons(row, model, miscon_embeddings, tokenizer)
    mini_l6_v2_preds.append(pred_list)
    actual.append(row['MisconceptionId'])
    if int(row['MisconceptionId']) in pred_list:
      contains_count += 1
  contains_ratio = contains_count / len(test_df)
  print(f'ratio of {model_name}\'s top 25 containing correct misconception: {contains_ratio}')
  apk_score = mapk(actual, mini_l6_v2_preds)
  print(f'mapk of {model_name}\'s top 25: {apk_score}')


evaluate_bi_encoder('sentence-transformers/all-MiniLM-L6-v2')
evaluate_bi_encoder('sentence-transformers/msmarco-MiniLM-L6-cos-v5')
evaluate_bi_encoder('sentence-transformers/paraphrase-MiniLM-L6-v2')


#fine-tune bi-encoder models
from sentence_transformers import InputExample, SentenceTransformer, losses, util
from torch.utils.data import DataLoader
from datasets import Dataset

train_examples = []
for _, row in train_df.iterrows():
    query_text = f"{row['SubjectName']}. {row['ConstructName']}. {row['QuestionText']} [SEP] {row['AnswerText']}"
    positive_example = row['MisconceptionName']

    train_examples.append(InputExample(texts=[query_text, positive_example], label=1.0))

    #hard negative: from same question but different answer - commented out as including this made the performance worse
    #hard_neg = train_df[(train_df['QuestionText'] == row['QuestionText']) & (train_df['MisconceptionName'] != row['MisconceptionName'])]
    #for _, hard_neg_row in hard_neg.iterrows():
        #hard_negative_example = hard_neg_row['MisconceptionName']
        #train_examples.append(InputExample(texts=[query_text, hard_negative_example], label=0.0))
        #print('appended hard neg')
    #random negative: from different questions
    rand_neg = train_df[(train_df['QuestionText'] != row['QuestionText']) & (train_df['MisconceptionName'] != row['MisconceptionName'])].sample(10)
    for _, rand_neg_row in rand_neg.iterrows():
        rand_negative_example = rand_neg_row['MisconceptionName']
        train_examples.append(InputExample(texts=[query_text, rand_negative_example], label=0.0))
        #print('appended rand neg')

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.CosineSimilarityLoss(model)

#train model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=20,
    warmup_steps=int(len(train_dataloader) * 0.1),
    show_progress_bar=True,
    optimizer_params={'lr': 1e-5}
)

#evaluate fine-tuned bi-encoder
def evaluate_fine_tuned_bi_encoder(model):
  contains_count = 0
  mini_l6_v2_preds = []
  actual = []
  misconceptions = miscon_df['MisconceptionName'].tolist()
  with torch.no_grad():
    misconception_embeddings = model.encode(misconceptions, convert_to_tensor=True, device='cuda')

  for i, row in test_df.iterrows():
    pred_list = bi_encoder_top_25_miscons(row, model, misconception_embeddings)
    mini_l6_v2_preds.append(pred_list)
    actual.append(row['MisconceptionId'])
    if int(row['MisconceptionId']) in pred_list:
      contains_count += 1
  contains_ratio = contains_count / len(test_df)
  print(f'ratio of mini_l6_v2 bi-encoder\'s top 25 containing correct misconception: {contains_ratio}')
  apk_score = mapk(actual, mini_l6_v2_preds)
  print(f'mapk of mini_l6_v2 bi-encoder\'s top 25: {apk_score}')
  return mini_l6_v2_preds

#save fine-tuned model
model.save('fine_tune_mini_l6_v5_model_30')
import shutil
shutil.make_archive('mini_l6_v5', 'zip', 'fine_tune_mini_l6_v5_model_30')

#Cross-Encoders
ms_macro_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
roberta_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/stsb-roberta-base')
deberta_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-small')

from transformers import AutoModelForSequenceClassification

def cross_encoder_rank_miscons(row, misconceptions, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    scores = []
    for misconception_idx in misconceptions:
        misconception_text = miscon_df['MisconceptionName'].tolist()[misconception_idx]
        query = f"[Question] {row['SubjectName'] + '. ' + row['ConstructName'] + '. ' + row['QuestionText']} [SEP] [Wrong Answer] {row['AnswerText']} [SEP] [Misconception] {misconception_text}"
        inputs = tokenizer(query, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits.item()
            scores.append((misconception_idx, score))

    ranked_misconceptions = sorted(scores, key=lambda x: x[1], reverse=True)
    return [item[0] for item in ranked_misconceptions]

from torch.nn.functional import softmax

def cross_encoder_logits_rank_miscons(row, misconceptions, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    scores = []
    for misconception_idx in misconceptions:
        misconception_text = miscon_df['MisconceptionName'].tolist()[misconception_idx]
        query = f"For {row['QuestionText']}, a math question on {row['SubjectName']} {row['ConstructName']}, the student {misconception_text}"
        candidate = f"The student derives wrong answer {row['AnswerText']}"
        input_query = f'[CLS] {query} [SEP] {candidate} [SEP]'
        print(input_query)

        inputs = tokenizer(input_query, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits)
            entailment_prob = probs[0][0].item()
            scores.append((misconception_idx, entailment_prob))

    ranked_misconceptions = sorted(scores, key=lambda x: x[1], reverse=True)
    return [item[0] for item in ranked_misconceptions]

def evaluate_cross_encoder(model_name, sample_size):
  model_preds = []
  actual = []
  for i, row in test_df.head(sample_size).iterrows():
    pred_list = bi_encoder_top_25_miscons(row, mini_l6_v2_model, mini_l6_v2_miscon_embeddings)
    #skip cross encoder if bi encoder res didn't even include correct answer to save run time
    if int(row['MisconceptionId']) in pred_list:
      if model_name == 'cross-encoder/nli-deberta-v3-small':
        cross_encoder_ranked = cross_encoder_rank_miscons(row, pred_list, model_name)
        print(cross_encoder_ranked)
      else:
        cross_encoder_ranked = cross_encoder_rank_miscons(row, pred_list, model_name)
      model_preds.append(cross_encoder_ranked)
    else:
      model_preds.append([-1])
    actual.append(int(row['MisconceptionId']))
  apk_score = mapk(actual, model_preds)
  print(f'mapk of mini_l6_v2 bi-encoder\'s top 25: {apk_score}')

evaluate_cross_encoder('cross-encoder/ms-marco-MiniLM-L-12-v2', 438)

evaluate_cross_encoder('cross-encoder/stsb-roberta-base', 438)

evaluate_cross_encoder('cross-encoder/nli-deberta-v3-small', 438)

#fine-tune cross-encoder with train examples
ce_train_examples = []
for _, row in train_df.iterrows():
    query_text = f"{row['SubjectName']}. {row['ConstructName']}. {row['QuestionText']} [SEP] {row['AnswerText']}"
    positive_example = row['MisconceptionName']

    ce_train_examples.append(InputExample(texts=[query_text, positive_example], label=1.0))

    #hard negative: from same question but different answer
    #hard_neg = train_df[(train_df['QuestionText'] == row['QuestionText']) & (train_df['MisconceptionName'] != row['MisconceptionName'])]
    #for _, hard_neg_row in hard_neg.iterrows():
        #hard_negative_example = hard_neg_row['MisconceptionName']
        #train_examples.append(InputExample(texts=[query_text, hard_negative_example], label=0.0))
    #random negative: from different questions
    rand_neg = train_df[(train_df['QuestionText'] != row['QuestionText']) & (train_df['MisconceptionName'] != row['MisconceptionName'])].sample(10)
    for _, rand_neg_row in rand_neg.iterrows():
        rand_negative_example = rand_neg_row['MisconceptionName']
        ce_train_examples.append(InputExample(texts=[query_text, rand_negative_example], label=0.0))

from sentence_transformers import CrossEncoder

train_examples = [
    InputExample(texts=[query, answer], label=score) for query, answer, score in train_data
]

ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', num_labels=1)
train_dataloader = DataLoader(ce_train_examples, shuffle=True, batch_size=16)

ce_model.fit(
    train_dataloader=train_dataloader,
    epochs=3,
    warmup_steps=int(len(train_dataloader) * 0.1)
)

#evaluate cross encoders
from transformers import AutoModelForSequenceClassification

def cross_encoder_rank_miscons(row, misconceptions, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    scores = []
    inputs = []
    for misconception_idx in misconceptions:
        misconception_text = miscon_df['MisconceptionName'].tolist()[misconception_idx]
        query = f"[Question] {row['SubjectName'] + '. ' + row['ConstructName'] + '. ' + row['QuestionText']} [SEP] [Wrong Answer] {row['AnswerText']}"
        candidate = f"[Misconception] {misconception_text}"
        inputs.append([query,candidate])

    with torch.no_grad():
        scores = ce_model.predict(inputs)
    misconception_score_pairs = list(zip(misconceptions, scores))
    top_25_pairs = sorted(misconception_score_pairs, key=lambda x: x[1], reverse=True)[:25]
    top_25_misconceptions = [misconception for misconception, score in top_25_pairs]

    return top_25_misconceptions

def evaluate_cross_encoder(model_name, sample_size):
  model_preds = []
  actual = []
  misconceptions = miscon_df['MisconceptionName'].tolist()
  with torch.no_grad():
    misconception_embeddings = ce_model.encode(misconceptions, convert_to_tensor=True, device='cuda')
  for i, row in test_df.iterrows():
    pred_list = bi_encoder_top_25_miscons(row, ce_model, misconception_embeddings)
    #skip cross encoder if bi encoder res didn't even include correct answer to save run time
    if int(row['MisconceptionId']) in pred_list:
      cross_encoder_ranked = cross_encoder_rank_miscons(row, pred_list, model_name)
      model_preds.append(cross_encoder_ranked)
    else:
      #print('not in list')
      model_preds.append([-1])
    actual.append(int(row['MisconceptionId']))
  apk_score = mapk(actual, model_preds)
  print(f'mapk of mini_l6_v2 bi-encoder\'s top 25: {apk_score}')