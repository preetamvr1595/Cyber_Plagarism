import os
import torch
import pandas as pd
import numpy as np
import joblib
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from scripts.perplexity_detector import PerplexityDetector
from scripts.stylometry_detector import StylometryDetector

class EnsembleDetector:
    def __init__(self, distilbert_path="models/distilbert_baseline", ensemble_model_path="models/ensemble_lr.pkl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load DistilBERT
        if os.path.exists(distilbert_path):
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_path)
            self.distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_path).to(self.device)
            self.distilbert_model.eval()
        else:
            print(f"Warning: DistilBERT not found at {distilbert_path}. Using base model.")
            self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            self.distilbert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(self.device)
            self.distilbert_model.eval()
            
        # Load other detectors
        self.perplexity_detector = PerplexityDetector()
        self.stylometry_detector = StylometryDetector()
        
        self.ensemble_model_path = ensemble_model_path
        self.ensemble_model = None
        
        if os.path.exists(self.ensemble_model_path):
            self.ensemble_model = joblib.load(self.ensemble_model_path)
            
    def get_features(self, text):
        """Extracts features from all sub-models."""
        # 1. DistilBERT feature (AI probability)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.distilbert_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            distilbert_ai_prob = probabilities[0][1].item()
            
        # 2. Perplexity feature
        perplexity = self.perplexity_detector.calculate_perplexity(text)
        
        # 3. Stylometry features
        stylometry_features = self.stylometry_detector.extract_features(text)
        
        features = [
            distilbert_ai_prob,
            perplexity,
            stylometry_features['avg_word_length'],
            stylometry_features['avg_sentence_length'],
            stylometry_features['type_token_ratio'],
            stylometry_features['punctuation_density']
        ]
        
        return features

    def train_ensemble(self, dataset_path="data/evaluation/eval_set.csv"):
        print(f"Training ensemble model using {dataset_path}...")
        df = pd.read_csv(dataset_path)
        
        # To save time, we might train on a subset if dataset is huge, but here we use all eval set
        X = []
        y = []
        
        print("Extracting features for ensemble training...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                feats = self.get_features(row['text'])
                X.append(feats)
                y.append(row['label'])
            except Exception as e:
                # skip bad rows
                continue
                
        X = np.array(X)
        y = np.array(y)
        
        # Train Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y)
        
        self.ensemble_model = lr
        os.makedirs(os.path.dirname(self.ensemble_model_path), exist_ok=True)
        joblib.dump(lr, self.ensemble_model_path)
        print(f"Ensemble model trained and saved to {self.ensemble_model_path}")
        
    def predict(self, text):
        feats = self.get_features(text)
        
        if self.ensemble_model is not None:
            # Scale features if needed, but simple LR might work fine
            X = np.array([feats])
            prob = self.ensemble_model.predict_proba(X)[0][1] # Probability of AI
            prediction = int(prob > 0.5)
        else:
            # Fallback heuristic if not trained
            distilbert_prob = feats[0]
            prediction = 1 if distilbert_prob > 0.5 else 0
            prob = distilbert_prob
            
        return {
            "prediction": "AI Generated" if prediction == 1 else "Human Written",
            "ai_probability": prob,
            "features": {
                "distilbert_ai_prob": feats[0],
                "perplexity": feats[1],
                "avg_word_length": feats[2],
                "avg_sentence_length": feats[3],
                "type_token_ratio": feats[4],
                "punctuation_density": feats[5]
            }
        }

if __name__ == "__main__":
    ensemble = EnsembleDetector()
    # Note: If no ensemble is trained, run ensemble.train_ensemble() first.
    res = ensemble.predict("This is a test document to verify the ensemble system works.")
    print(res)
