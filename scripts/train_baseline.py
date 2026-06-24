import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

def train_baseline():
    print("Loading dataset...")
    dataset_path = "data/balanced_ai_detection_dataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run collect_dataset.py first.")
        
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["text", "label"])
    
    # Stratified Train/Test Split (80% train, 20% test)
    train_df, eval_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )
    
    # Save the eval dataset for later comprehensive evaluation (Tasks 7, 14)
    os.makedirs("data/evaluation", exist_ok=True)
    eval_df.to_csv("data/evaluation/eval_set.csv", index=False)
    
    # Convert to HF Dataset
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    # Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )
        
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    
    train_dataset = train_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.rename_column("label", "labels")
    
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./outputs/checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./outputs/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the best model
    os.makedirs("models/distilbert_baseline", exist_ok=True)
    model.save_pretrained("models/distilbert_baseline")
    tokenizer.save_pretrained("models/distilbert_baseline")
    
    print("Training Complete! Model saved to models/distilbert_baseline")

if __name__ == "__main__":
    train_baseline()
