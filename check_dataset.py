import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv("data/ai_detection_dataset.csv")

print(df.head())

print("\nShape:", df.shape)

print("\nLabels:")
print(df["label"].value_counts()) 