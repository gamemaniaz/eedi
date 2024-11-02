import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os

DATA_PATH = 'dataset'
MODEL_NAME = 'BAAI/bge-large-en-v1.5'
RETRIEVE_NUM = 200
ITERATIONS = 5

BATCH_SIZE = 64
EPOCHS = 2
LR = 2e-5
GRAD_ACC_STEPS = 1

MODEL_OUTPUT_PATH = 'model/'

train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
misconception_mapping = pd.read_csv(f"{DATA_PATH}/misconception_mapping.csv")

common_cols = [
    "QuestionId",
    "ConstructName",
    "SubjectName",
    "QuestionText",
    "CorrectAnswer",
]

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

train_long = train_long.merge(train_misconception_long, on="QuestionId_Answer", how="inner")

train_long["AllText"] = train_long["ConstructName"] + "#####" + train_long["SubjectName"] + "#####" + train_long["QuestionText"] + "#####" + train_long["AnswerText"]

# Merge to get MisconceptionName
train_long = train_long.merge(misconception_mapping, on="MisconceptionId", how="left", suffixes=('', '_Misconception'))

# Initialize the initial model
model = SentenceTransformer(MODEL_NAME)

# Prepare misconception mapping
misconception_names = misconception_mapping["MisconceptionName"].tolist()
misconception_ids = misconception_mapping["MisconceptionId"].tolist()
misconception_id_to_index = {misconception_ids[i]: i for i in range(len(misconception_ids))}

# Training iterations
for iteration in range(ITERATIONS):
    print(f"Starting iteration {iteration+1}/{ITERATIONS}")

    # Compute embeddings for AllText
    print("Computing embeddings for AllText...")
    alltext_embeddings = model.encode(
        train_long["AllText"].tolist(),
        batch_size=128,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Compute embeddings for MisconceptionName
    print("Computing embeddings for MisconceptionName...")
    misconception_embeddings = model.encode(
        misconception_names,
        batch_size=128,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
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
        topk_indices = torch.topk(similarities, k=RETRIEVE_NUM).indices.tolist()

        # For each hard negative, create an InputExample
        for hard_neg_index in topk_indices:
            hard_neg_misconception_name = misconception_names[hard_neg_index]
            positive_misconception_name = train_long.loc[idx, "MisconceptionName"]
            alltext = train_long.loc[idx, "AllText"]

            input_examples.append(InputExample(
                texts=[alltext, positive_misconception_name, hard_neg_misconception_name]
            ))

    # Set up DataLoader
    train_dataloader = DataLoader(input_examples, shuffle=True, batch_size=BATCH_SIZE)

    # Set up loss function
    train_loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.5)

    # Calculate warmup steps
    warmup_steps = int(len(train_dataloader) * EPOCHS * 0.1)  # 10% of training steps

    # Fine-tune the model
    print(f"Fine-tuning the model for iteration {iteration+1}...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        optimizer_params={'lr': LR},
        warmup_steps=warmup_steps,
        output_path=None,
        use_amp=True,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        show_progress_bar=True
    )

    # Save the model after each iteration
    iteration_model_path = os.path.join(MODEL_OUTPUT_PATH, f"model_iteration_{iteration+1}")
    model.save(iteration_model_path)
    print(f"Model saved for iteration {iteration+1} at {iteration_model_path}")

    # Update the model variable to use the latest model in the next iteration
    model = SentenceTransformer(iteration_model_path)

print("Training completed.")
