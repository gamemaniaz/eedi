import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from tqdm import tqdm
import os

# Set paths and parameters
DATA_PATH = 'dataset'
CACHE_DIR = 'pretrained_model/'
EMBEDDING_MODEL_NAME = 'rombodawg/Rombos-LLM-V2.5-Qwen-72b'
RERANK_MODEL_NAME = 'upstage/SOLAR-10.7B-Instruct-v1.0'
RETRIEVE_NUM_HARD_NEGATIVES = 200  # Number of hard negatives for embedding model training
RETRIEVE_NUM_RERANK = 100  # Number of hard negatives for rerank model training
ITERATIONS = 5

BATCH_SIZE = 12
EPOCHS = 2  # Number of epochs in each iteration
LR = 1e-5  # Learning rate suitable for large models

MODEL_OUTPUT_PATH = 'model/'

# Ensure that CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read data
train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
misconception_mapping = pd.read_csv(f"{DATA_PATH}/misconception_mapping.csv")

# Process data
common_cols = [
    "QuestionId",
    "ConstructName",
    "SubjectName",
    "QuestionText",
    "CorrectAnswer",
]

# Unpivot the AnswerText columns
train_long = (
    train_df[common_cols + [f"Answer{alpha}Text" for alpha in ["A", "B", "C", "D"]]]
    .melt(
        id_vars=common_cols,
        value_vars=[f"Answer{alpha}Text" for alpha in ["A", "B", "C", "D"]],
        var_name="AnswerType",
        value_name="AnswerText"
    )
)
train_long["AnswerAlphabet"] = train_long["AnswerType"].str.extract(r"Answer([A-D])Text$")
train_long["QuestionId_Answer"] = train_long["QuestionId"].astype(str) + "_" + train_long["AnswerAlphabet"]

# Unpivot the MisconceptionId columns
train_misconception_long = (
    train_df[common_cols + [f"Misconception{alpha}Id" for alpha in ["A", "B", "C", "D"]]]
    .melt(
        id_vars=common_cols,
        value_vars=[f"Misconception{alpha}Id" for alpha in ["A", "B", "C", "D"]],
        var_name="MisconceptionType",
        value_name="MisconceptionId"
    )
)
train_misconception_long["AnswerAlphabet"] = train_misconception_long["MisconceptionType"].str.extract(r"Misconception([A-D])Id$")
train_misconception_long["QuestionId_Answer"] = train_misconception_long["QuestionId"].astype(str) + "_" + train_misconception_long["AnswerAlphabet"]
train_misconception_long = train_misconception_long[["QuestionId_Answer", "MisconceptionId"]].dropna()

# Merge to get the MisconceptionId for each AnswerText
train_long = train_long.merge(train_misconception_long, on="QuestionId_Answer", how="inner")

# Create AllText by concatenating ConstructName, SubjectName, QuestionText, and AnswerText
train_long["AllText"] = train_long["ConstructName"] + "#####" + train_long["SubjectName"] + "#####" + train_long["QuestionText"] + "#####" + train_long["AnswerText"]

# Merge to get MisconceptionName
train_long = train_long.merge(misconception_mapping, on="MisconceptionId", how="left", suffixes=('', '_Misconception'))

# Prepare misconception mapping
misconception_names = misconception_mapping["MisconceptionName"].tolist()
misconception_ids = misconception_mapping["MisconceptionId"].tolist()
misconception_id_to_index = {misconception_ids[i]: i for i in range(len(misconception_ids))}

# Define custom dataset
class ContrastiveDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        inputs = self.tokenizer(
            ex['anchor'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        positives = self.tokenizer(
            ex['positive'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        negatives = self.tokenizer(
            ex['negative'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'anchor_input_ids': inputs['input_ids'].squeeze(0),
            'anchor_attention_mask': inputs['attention_mask'].squeeze(0),
            'positive_input_ids': positives['input_ids'].squeeze(0),
            'positive_attention_mask': positives['attention_mask'].squeeze(0),
            'negative_input_ids': negatives['input_ids'].squeeze(0),
            'negative_attention_mask': negatives['attention_mask'].squeeze(0),
        }

# Function to compute embeddings
def compute_embeddings(model, tokenizer, texts, batch_size=8):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            outputs = model(**inputs, return_dict=True)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

# Training iterations
for iteration in range(ITERATIONS):
    print(f"\n=== Starting iteration {iteration+1}/{ITERATIONS} ===\n")

    # Step 1: Use current embedding model to mine hard negatives
    if iteration == 0:
        # Load initial embedding model m0_0
        embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, cache_dir=CACHE_DIR)
        embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, cache_dir=CACHE_DIR).to(device)
    else:
        # Load the fine-tuned embedding model from previous iteration
        embedding_model_path = os.path.join(MODEL_OUTPUT_PATH, f"embedding_model_iteration_{iteration}")
        embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
        embedding_model = AutoModel.from_pretrained(embedding_model_path).to(device)

    print("Computing embeddings for AllText...")
    alltext_embeddings = compute_embeddings(
        embedding_model,
        embedding_tokenizer,
        train_long["AllText"].tolist(),
        batch_size=BATCH_SIZE
    )

    print("Computing embeddings for MisconceptionName...")
    misconception_embeddings = compute_embeddings(
        embedding_model,
        embedding_tokenizer,
        misconception_names,
        batch_size=BATCH_SIZE
    )

    # Compute cosine similarities
    print("Computing cosine similarities...")
    cosine_similarities = torch.mm(alltext_embeddings, misconception_embeddings.T)

    # Prepare InputExamples with hard negatives
    input_examples = []
    print("Mining hard negatives and preparing training examples...")
    for idx in tqdm(range(len(train_long))):
        correct_misconception_id = train_long.loc[idx, "MisconceptionId"]
        correct_misconception_index = misconception_id_to_index[correct_misconception_id]

        similarities = cosine_similarities[idx].clone()
        similarities[correct_misconception_index] = -1e6  # Exclude the correct one

        # Get top-k hard negatives
        topk_indices = torch.topk(similarities, k=RETRIEVE_NUM_HARD_NEGATIVES).indices.tolist()

        # For each hard negative, create an example
        for hard_neg_index in topk_indices:
            hard_neg_misconception_name = misconception_names[hard_neg_index]
            positive_misconception_name = train_long.loc[idx, "MisconceptionName"]
            alltext = train_long.loc[idx, "AllText"]

            input_examples.append({
                'anchor': alltext,
                'positive': positive_misconception_name,
                'negative': hard_neg_misconception_name
            })

    # Step 2: Fine-tune embedding model with contrastive learning
    print(f"Fine-tuning the embedding model for iteration {iteration+1}...")
    # Prepare dataset and dataloader
    contrastive_dataset = ContrastiveDataset(input_examples, embedding_tokenizer)
    contrastive_dataloader = DataLoader(contrastive_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(embedding_model.parameters(), lr=LR)
    embedding_model.train()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for batch in tqdm(contrastive_dataloader):
            optimizer.zero_grad()
            anchor_inputs = {
                'input_ids': batch['anchor_input_ids'].to(device),
                'attention_mask': batch['anchor_attention_mask'].to(device),
            }
            positive_inputs = {
                'input_ids': batch['positive_input_ids'].to(device),
                'attention_mask': batch['positive_attention_mask'].to(device),
            }
            negative_inputs = {
                'input_ids': batch['negative_input_ids'].to(device),
                'attention_mask': batch['negative_attention_mask'].to(device),
            }
            anchor_output = embedding_model(**anchor_inputs, return_dict=True).last_hidden_state[:, 0, :]
            positive_output = embedding_model(**positive_inputs, return_dict=True).last_hidden_state[:, 0, :]
            negative_output = embedding_model(**negative_inputs, return_dict=True).last_hidden_state[:, 0, :]
            # Normalize embeddings
            anchor_output = F.normalize(anchor_output, p=2, dim=1)
            positive_output = F.normalize(positive_output, p=2, dim=1)
            negative_output = F.normalize(negative_output, p=2, dim=1)
            # Compute triplet loss
            pos_sim = F.cosine_similarity(anchor_output, positive_output)
            neg_sim = F.cosine_similarity(anchor_output, negative_output)
            loss = F.relu(1 - pos_sim + neg_sim).mean()
            loss.backward()
            optimizer.step()

    # Save the fine-tuned embedding model
    embedding_model_path = os.path.join(MODEL_OUTPUT_PATH, f"embedding_model_iteration_{iteration+1}")
    embedding_model.save_pretrained(embedding_model_path)
    embedding_tokenizer.save_pretrained(embedding_model_path)
    print(f"Embedding model saved for iteration {iteration+1} at {embedding_model_path}")

    # Step 3: Use the updated embedding model to get hard negatives for rerank model
    print("Computing embeddings with updated embedding model for rerank training...")
    alltext_embeddings = compute_embeddings(
        embedding_model,
        embedding_tokenizer,
        train_long["AllText"].tolist(),
        batch_size=BATCH_SIZE
    )

    misconception_embeddings = compute_embeddings(
        embedding_model,
        embedding_tokenizer,
        misconception_names,
        batch_size=BATCH_SIZE
    )

    cosine_similarities = torch.mm(alltext_embeddings, misconception_embeddings.T)

    # Prepare InputExamples with hard negatives for rerank model
    rerank_examples = []
    print("Preparing training examples for rerank model...")
    for idx in tqdm(range(len(train_long))):
        correct_misconception_id = train_long.loc[idx, "MisconceptionId"]
        correct_misconception_index = misconception_id_to_index[correct_misconception_id]

        similarities = cosine_similarities[idx].clone()
        similarities[correct_misconception_index] = -1e6  # Exclude the correct one

        # Get top-k hard negatives
        topk_indices = torch.topk(similarities, k=RETRIEVE_NUM_RERANK).indices.tolist()

        # For each hard negative, create an example
        for hard_neg_index in topk_indices:
            hard_neg_misconception_name = misconception_names[hard_neg_index]
            positive_misconception_name = train_long.loc[idx, "MisconceptionName"]
            alltext = train_long.loc[idx, "AllText"]

            rerank_examples.append({
                'query': alltext,
                'positive': positive_misconception_name,
                'negative': hard_neg_misconception_name
            })

    # Step 4: Fine-tune rerank model
    print(f"Fine-tuning the rerank model for iteration {iteration+1}...")
    # Load rerank model (SOLAR-10.7B-Instruct-v1.0)
    # if iteration == 0:
    rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME, cache_dir=CACHE_DIR)
    rerank_model = AutoModel.from_pretrained(RERANK_MODEL_NAME, cache_dir=CACHE_DIR).to(device)
    # else:
    #     rerank_model_path = os.path.join(MODEL_OUTPUT_PATH, f"rerank_model_iteration_{iteration}")
    #     rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_path)
    #     rerank_model = AutoModel.from_pretrained(rerank_model_path).to(device)

    # Prepare dataset and dataloader
    rerank_dataset = ContrastiveDataset(rerank_examples, rerank_tokenizer)
    rerank_dataloader = DataLoader(rerank_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(rerank_model.parameters(), lr=LR)
    rerank_model.train()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for batch in tqdm(rerank_dataloader):
            optimizer.zero_grad()
            query_inputs = {
                'input_ids': batch['anchor_input_ids'].to(device),
                'attention_mask': batch['anchor_attention_mask'].to(device),
            }
            positive_inputs = {
                'input_ids': batch['positive_input_ids'].to(device),
                'attention_mask': batch['positive_attention_mask'].to(device),
            }
            negative_inputs = {
                'input_ids': batch['negative_input_ids'].to(device),
                'attention_mask': batch['negative_attention_mask'].to(device),
            }
            query_output = rerank_model(**query_inputs, return_dict=True).last_hidden_state[:, 0, :]
            positive_output = rerank_model(**positive_inputs, return_dict=True).last_hidden_state[:, 0, :]
            negative_output = rerank_model(**negative_inputs, return_dict=True).last_hidden_state[:, 0, :]
            # Normalize embeddings
            query_output = F.normalize(query_output, p=2, dim=1)
            positive_output = F.normalize(positive_output, p=2, dim=1)
            negative_output = F.normalize(negative_output, p=2, dim=1)
            # Compute triplet loss
            pos_sim = F.cosine_similarity(query_output, positive_output)
            neg_sim = F.cosine_similarity(query_output, negative_output)
            loss = F.relu(1 - pos_sim + neg_sim).mean()
            loss.backward()
            optimizer.step()

    # Save the fine-tuned rerank model
    rerank_model_path = os.path.join(MODEL_OUTPUT_PATH, f"rerank_model_iteration_{iteration+1}")
    rerank_model.save_pretrained(rerank_model_path)
    rerank_tokenizer.save_pretrained(rerank_model_path)
    print(f"Rerank model saved for iteration {iteration+1} at {rerank_model_path}")

    print(f"=== Iteration {iteration+1} completed ===\n")

print("Training completed.")
