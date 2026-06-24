import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

file_path = "data/open_qa.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    first_line = f.readline()

data = json.loads(first_line)

print(data)