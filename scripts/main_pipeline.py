import os
import sys
import json
from scripts.document_processor import process_document
from scripts.ensemble_detector import EnsembleDetector

def run_pipeline(file_path):
    print(f"Processing document: {file_path}")
    
    try:
        text = process_document(file_path)
    except Exception as e:
        print(f"Error reading document: {e}")
        return
        
    print(f"Extracted {len(text)} characters of text.")
    if len(text.strip()) < 10:
        print("Warning: Document is empty or too short for reliable detection.")
        
    print("Running Ensemble AI Detector...")
    detector = EnsembleDetector()
    
    try:
        result = detector.predict(text)
    except Exception as e:
        print(f"Error during detection: {e}")
        return
        
    print("\n" + "="*40)
    print(" " * 10 + "SCHOLARSHIELD REPORT")
    print("="*40)
    print(f"Prediction       : {result['prediction']}")
    print(f"AI Probability   : {result['ai_probability']*100:.2f}%")
    print("\n--- Model Breakdown ---")
    print(f"DistilBERT Prob  : {result['features']['distilbert_ai_prob']*100:.2f}%")
    print(f"GPT-2 Perplexity : {result['features']['perplexity']:.2f} (Lower often means AI)")
    print(f"Avg Word Length  : {result['features']['avg_word_length']:.2f}")
    print(f"Type/Token Ratio : {result['features']['type_token_ratio']:.2f} (Higher means richer vocabulary)")
    print("="*40)
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.main_pipeline <path_to_document>")
        file_path = input("Enter path to document (TXT, DOCX, PDF): ").strip().strip('"').strip("'")
        if file_path:
            run_pipeline(file_path)
    else:
        run_pipeline(sys.argv[1])
