import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv("data/ai_detection_dataset.csv")

human = df[df["label"] == 0]
ai = df[df["label"] == 1]

# Take same number of AI samples as human samples
ai_balanced = ai.sample(n=len(human), random_state=42)

balanced_df = pd.concat([human, ai_balanced])

balanced_df = balanced_df.sample(frac=1, random_state=42)

print(balanced_df["label"].value_counts())

balanced_df.to_csv(
    "data/balanced_ai_detection_dataset.csv",
    index=False
)

print("Balanced dataset saved!")
