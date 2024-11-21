#select top k questions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def get_top_25_relevant(row, processed_df, column_name):
    same_subject = []
    if len(same_subject) < 25:
        same_subject.extend(processed_df[processed_df['ConstructName'] == row['ConstructName']][column_name].tolist())

    if len(same_subject) < 25:
        all_texts = processed_df['QuestionText'].tolist() + [row['QuestionText']]
        vectorizer = TfidfVectorizer().fit_transform(all_texts)
        similarity_matrix = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()

        similar_indices = similarity_matrix.argsort()[::-1]
        similar_misconceptions = processed_df.iloc[similar_indices][column_name].tolist()

        for mis_id in similar_misconceptions:
            if len(same_subject) >= 15:
                break
            if mis_id not in same_subject:
                same_subject.append(mis_id)

    if len(same_subject) < 25:
        same_construct = processed_df[processed_df['SubjectName'] == row['SubjectName']][column_name].tolist()
        same_subject.extend([mis_id for mis_id in same_construct if mis_id not in same_subject])

    return same_subject[:25]

def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

#process dataframes
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

  print(df.head())
  return df

import pandas as pd
test_df = pd.read_csv('test_data.csv')
train_df = pd.read_csv('train_data.csv')
miscon_df = pd.read_csv('misconception_mapping.csv')
processed_train_df = process_df(train_df)
processed_test_df = process_df(test_df)


#construct prompt
def construct_few_shot_prompt(filtered_df, test_row, max_examples=25):
    prompt = "Below are examples of questions with subjects, constructs, answers, and misconceptions.\n\n"
    i = 0
    for idx, row in filtered_df.head(max_examples).iterrows():
        i += 1
        example = f"""Example {i}:
Question: {row['QuestionText']}
Answer: {row['AnswerText']}
Misconception: {row['MisconceptionName']}\n\n"""
        prompt += example

    prompt += f"""Given the following information, generate a numbered list of 5 misconceptions based on examples:

Question: {test_row['QuestionText']}
Answer: {test_row['AnswerText']}
### Misconception:
"""

    return prompt

#construct alternative prompt (this was another experiment)
def construct_few_shot_prompt2(filtered_df, test_row, max_examples=3):
    prompt = "Based on the following maths question on " + test_row['SubjectName']
    prompt += ", choose one from the misconception list that causes the incorrect student answer"
    qn = test_row['QuestionText'].replace('\n\n\n', ' ').replace('\n', ' ')
    prompt += f"""
Question: {qn}
Incorrect student answer: {test_row['AnswerText']}\n\n"""
    prompt += "Misconception list:\n"
    i = 0
    for idx, row in filtered_df.head(max_examples).iterrows():
        i+=1
        prompt += f"""{i}. {row['MisconceptionName']}\n"""
    prompt += "### Misconception:"
    return prompt


#load LLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
import os
device="cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
model_name =  "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.to(device)
model.gradient_checkpointing_enable()

def generate_response(input_text):
  inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)

  with torch.no_grad():
      outputs = model.generate(inputs.input_ids, max_length=2048, temperature=0.5)#, top_k=50, top_p=0.9)

  response = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return response


#helper to match LLM response to misconception
def get_misconception_ids(misconceptions):
    matched_ids = []

    for miscon in misconceptions:
        exact_match = miscon_df[miscon_df['MisconceptionName'] == miscon]
        if not exact_match.empty:
            matched_ids.append(exact_match['MisconceptionId'].values[0])
        else:
            all_names = miscon_df['MisconceptionName'].tolist() + [miscon]
            vectorizer = TfidfVectorizer().fit_transform(all_names)
            similarity_matrix = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()
            most_similar_index = similarity_matrix.argmax()
            matched_ids.append(miscon_df.iloc[most_similar_index]['MisconceptionId'])

    return remove_duplicates(matched_ids)


#prediction pipeline
import re
def predict_pipeline(test_row):
  #step 1: find example questions
  qn_ids = get_top_25_relevant(test_row, processed_train_df, 'QuestionId')
  qn_df = processed_train_df[processed_train_df['QuestionId'].isin(qn_ids)]

  #step 2: construct prompt
  prompt = construct_few_shot_prompt(qn_df, test_row)

  #step 3: generate response
  response = generate_response(prompt)
  
  #step 4: parse response
  pattern = r"\d+\.\s+(.*)"
  misconceptions = re.findall(pattern, response)
  misconceptions = remove_duplicates(misconceptions)
  #retry generation once if not successful
  if len(misconceptions) < 3:
    response = generate_response(prompt)
    misconceptions = re.findall(pattern, response)
    misconceptions = remove_duplicates(misconceptions)
  
  #step 5: map misconception response back to id
  result = get_misconception_ids(misconceptions)

  #step 6: augment the list with 20 more guesses
  relevant_ids = get_top_25_relevant(test_row, processed_train_df, 'MisconceptionId')

  for rel_id in relevant_ids:
    if len(result) >= 25:
      break
    if rel_id not in result:
      result.append(rel_id)
  return result

#the result of this was 0.0790