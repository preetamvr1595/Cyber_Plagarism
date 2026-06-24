import json
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

file_path = "data/open_qa.jsonl"

rows = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        # Human Answers
        for answer in data.get("human_answers", []):
            rows.append({
                "text": answer,
                "label": 0
            })

        # ChatGPT Answers
        for answer in data.get("chatgpt_answers", []):
            rows.append({
                "text": answer,
                "label": 1
            })

df = pd.DataFrame(rows)

print(df.head())

print("\nTotal Samples:", len(df))

df.to_csv(
    "data/ai_detection_dataset.csv",
    index=False
)

print("Dataset Saved Successfully!")