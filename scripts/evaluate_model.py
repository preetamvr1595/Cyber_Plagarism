import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from scripts.ensemble_detector import EnsembleDetector

def evaluate():
    print("Loading evaluation dataset...")
    eval_path = "data/evaluation/eval_set.csv"
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Evaluation dataset not found at {eval_path}.")
        
    df = pd.read_csv(eval_path)
    # We might want to evaluate on a subset if it's too large, but let's do up to 500 for a quick evaluation
    if len(df) > 500:
        df = df.sample(500, random_state=42).reset_index(drop=True)
        
    detector = EnsembleDetector()
    
    # If the ensemble model isn't trained yet, train it on the rest of the eval set
    if detector.ensemble_model is None:
        print("Ensemble model not trained. Training now...")
        detector.train_ensemble(eval_path)
        
    y_true = []
    y_pred = []
    failure_cases = []
    
    print("Evaluating...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        label = row['label']
        
        try:
            res = detector.predict(text)
            pred = 1 if res['prediction'] == "AI Generated" else 0
            
            y_true.append(label)
            y_pred.append(pred)
            
            if pred != label:
                failure_cases.append({
                    "text": text,
                    "true_label": label,
                    "predicted_label": pred,
                    "ai_probability": res['ai_probability']
                })
        except Exception as e:
            continue
            
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    # Save Evaluation Report CSV
    os.makedirs("outputs", exist_ok=True)
    report_df = pd.DataFrame([{
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }])
    report_df.to_csv("outputs/evaluation_report.csv", index=False)
    
    # Save Confusion Matrix PNG
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    
    # Save Performance Summary PDF
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    summary_text = (
        f"ScholarShield AI Detection Performance\n\n"
        f"Accuracy:  {acc*100:.2f}%\n"
        f"Precision: {prec*100:.2f}%\n"
        f"Recall:    {rec*100:.2f}%\n"
        f"F1 Score:  {f1*100:.2f}%\n\n"
        f"Total evaluated samples: {len(y_true)}\n"
        f"Total failure cases: {len(failure_cases)}\n"
    )
    ax.text(0.1, 0.5, summary_text, fontsize=14, family='monospace', va='center')
    plt.savefig("outputs/performance_summary.pdf")
    
    # Save Failure Cases
    if failure_cases:
        fail_df = pd.DataFrame(failure_cases)
        fail_df.to_csv("data/failure_cases.csv", index=False)
        print(f"Saved {len(failure_cases)} failure cases to data/failure_cases.csv")
        
    print("Evaluation complete. Outputs saved to outputs/ directory.")

if __name__ == "__main__":
    evaluate()
