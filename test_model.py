import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification

# Load Model
model_path = "./scholarshield_ai_detector"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Test Text
text = input("Enter text: ")

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=256
)

with torch.no_grad():
    outputs = model(**inputs)

prediction = torch.argmax(outputs.logits, dim=1)

if prediction.item() == 1:
    print("\nAI Generated Text")
else:
    print("\nHuman Written Text")