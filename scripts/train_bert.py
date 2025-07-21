import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import os
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

# Constants
MODEL_NAME = "bert-base-uncased"
DATA_PATH = "data/processed/train_clean.csv"

# Custom Dataset
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Load data
df = pd.read_csv(DATA_PATH)
texts = df["text"].tolist()
labels = df["target"].tolist()

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_dataset = TweetDataset(X_train, y_train, tokenizer)
val_dataset = TweetDataset(X_val, y_val, tokenizer)

# MLflow setup
mlflow.set_tracking_uri("file:./mlruns")  # local folder
mlflow.set_experiment("bert_disaster_tweets1")

with mlflow.start_run(run_name="bert_base_uncased"):
    # Define model
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: {
            "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1)),
            "f1": f1_score(p.label_ids, p.predictions.argmax(-1))
        }
    )

    trainer.train()
    metrics = trainer.evaluate()

    mlflow.log_params(training_args.to_dict())
    mlflow.log_metrics(metrics)

    # Save model
    model_path = "models/bert"
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    mlflow.log_artifacts(model_path, artifact_path="bert_model")

sample_input = tokenizer("This is a disaster", return_tensors="pt")
signature = infer_signature(sample_input["input_ids"].numpy(), model(**sample_input).logits.detach().numpy())

mlflow.pytorch.log_model(model, "bert_model", signature=signature)
