import fitz
import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification

# Load Model
model_path = "./scholarshield_ai_detector"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

pdf_path = input("Enter PDF Path: ").strip().strip('"').strip("'")

# Check if file exists, and handle spacing/case mismatches
import os
if not os.path.exists(pdf_path):
    dir_name = os.path.dirname(pdf_path)
    base_name = os.path.basename(pdf_path)
    if not dir_name:
        dir_name = "."
    if os.path.exists(dir_name):
        normalized_base = " ".join(base_name.split()).lower()
        for f in os.listdir(dir_name):
            if " ".join(f.split()).lower() == normalized_base:
                pdf_path = os.path.join(dir_name, f)
                print(f"Auto-resolved path to: {pdf_path}")
                break

# Extract Text
doc = fitz.open(pdf_path)

text = ""

for page in doc:
    text += page.get_text()

# Clean text (replace PUA bullet points to avoid tokenizer mismatches)
text = text.replace('\uf0b7', '•')

# Limit text for first test
text = text[:3000]

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=256
)

with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.softmax(outputs.logits, dim=1)

human_prob = probabilities[0][0].item() * 100
ai_prob = probabilities[0][1].item() * 100

print("\n----- RESULT -----")

print(f"Human Probability: {human_prob:.2f}%")
print(f"AI Probability: {ai_prob:.2f}%")

if ai_prob > human_prob:
    print("Prediction: AI Generated")
else:
    print("Prediction: Human Written")