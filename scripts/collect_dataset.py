import os
import pandas as pd
from datasets import load_dataset
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def collect_and_build_dataset():
    print("Downloading HC3 dataset (Human vs ChatGPT)...")
    # HC3 dataset contains 'human_answers' and 'chatgpt_answers'
    dataset = load_dataset("Hello-SimpleAI/HC3", "all", split="train")
    
    human_texts = []
    ai_texts = []
    
    # We will sample 5000 of each to keep the baseline fast and balanced
    sample_size = 5000
    
    print("Extracting texts...")
    for row in dataset:
        if len(human_texts) >= sample_size and len(ai_texts) >= sample_size:
            break
            
        if len(human_texts) < sample_size and row['human_answers']:
            human_texts.append(clean_text(row['human_answers'][0]))
            
        if len(ai_texts) < sample_size and row['chatgpt_answers']:
            ai_texts.append(clean_text(row['chatgpt_answers'][0]))
            
    print(f"Collected {len(human_texts)} human texts and {len(ai_texts)} AI texts.")
    
    # Create DataFrame
    human_df = pd.DataFrame({'text': human_texts, 'label': 0}) # 0 for Human
    ai_df = pd.DataFrame({'text': ai_texts, 'label': 1}) # 1 for AI
    
    # Merge and shuffle
    combined_df = pd.concat([human_df, ai_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Deduplicate
    initial_len = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['text'])
    print(f"Removed {initial_len - len(combined_df)} duplicates.")
    
    # Remove empty or very short strings
    combined_df = combined_df[combined_df['text'].str.len() > 50]
    print(f"Final dataset size: {len(combined_df)} rows.")
    
    # Balance classes exactly
    min_class_size = combined_df['label'].value_counts().min()
    balanced_df = combined_df.groupby('label').head(min_class_size)
    
    print(f"Balanced dataset size: {len(balanced_df)} ({min_class_size} per class)")
    
    # Save to disk
    os.makedirs("data", exist_ok=True)
    balanced_df.to_csv("data/balanced_ai_detection_dataset.csv", index=False)
    
    # Also save to training directories for structure
    os.makedirs("data/training/human", exist_ok=True)
    os.makedirs("data/training/ai", exist_ok=True)
    
    print("Saving text files to directory structure...")
    for idx, row in balanced_df.iterrows():
        label_dir = "ai" if row['label'] == 1 else "human"
        filename = f"data/training/{label_dir}/sample_{idx}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(row['text'])
        except Exception as e:
            pass # Skip problematic encodings if any
            
    print("Dataset collection and building complete!")

if __name__ == "__main__":
    collect_and_build_dataset()
