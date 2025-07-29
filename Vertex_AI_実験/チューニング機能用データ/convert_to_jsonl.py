import csv
import json
import random

# 入力CSVと出力JSONLのパス
csv_path = "map_data/train.csv"
train_jsonl_path = "Vertex_AI_実験/sft_train_data_train.jsonl"
valid_jsonl_path = "Vertex_AI_実験/sft_train_data_valid.jsonl"
valid_size = 5000

# データをすべて読み込む
with open(csv_path, encoding="utf-8") as csvfile:
    reader = list(csv.DictReader(csvfile))
    random.shuffle(reader)
    valid_rows = reader[:valid_size]
    train_rows = reader[valid_size:]


def row_to_jsonl(row):
    user_text = (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Student's explanation: {row['StudentExplanation']}"
    )
    model_text = row["Category"]
    return {
        "contents": [
            {"role": "user", "parts": [{"text": user_text}]},
            {"role": "model", "parts": [{"text": model_text}]},
        ]
    }


with open(train_jsonl_path, "w", encoding="utf-8") as train_file:
    for row in train_rows:
        train_file.write(json.dumps(row_to_jsonl(row), ensure_ascii=False) + "\n")

with open(valid_jsonl_path, "w", encoding="utf-8") as valid_file:
    for row in valid_rows:
        valid_file.write(json.dumps(row_to_jsonl(row), ensure_ascii=False) + "\n")
