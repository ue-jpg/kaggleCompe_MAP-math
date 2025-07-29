import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 設定
MODEL_NAME = "nvidia/AceMath-7B-Instruct"
NUM_LABELS = 65  # ラベル数に合わせて変更
MAX_LENGTH = 1024
NEW_MODEL_NAME = "acemath-7b-qlora-improved"
DATA_PATH = "./train.csv"  # 訓練データのパス

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 使用デバイス: {device}")

# データ読み込み
train_df = pd.read_csv(DATA_PATH)
all_labels = (
    train_df["Category"].astype(str) + ":" + train_df["Misconception"].fillna("NA").astype(str)
).unique()
all_labels = sorted(all_labels)
train_df["full_label"] = (
    train_df["Category"].astype(str) + ":" + train_df["Misconception"].fillna("NA").astype(str)
)

# 出現回数が1のラベルを特定し、それらの行を除外
label_counts = train_df["full_label"].value_counts()
single_occurrence_labels = label_counts[label_counts == 1].index
train_df = train_df[~train_df["full_label"].isin(single_occurrence_labels)]

# プロンプト生成関数
def get_improved_compact_prompt(question, answer, explanation, all_labels):
    labels_text = "\n".join([f"- {label}" for label in all_labels])
    prompt = f"""You are an expert math educator analyzing student responses for mathematical misconceptions.

Question: {question}
Student's Answer: {answer}
Student's Explanation: {explanation}

CLASSIFICATION GUIDELINES:
• True_Correct:NA = Student demonstrates correct understanding
• False_Correct:NA = Student gives correct answer but for wrong reasons
• True_Neither:NA = Correct answer but unclear/incomplete reasoning
• False_Neither:NA = Incorrect answer but no specific misconception identified
• True_Misconception:[Type] = Correct answer but demonstrates specific misconception
• False_Misconception:[Type] = Incorrect answer with identifiable misconception

TASK: Classify this student's response using EXACTLY ONE of these {len(all_labels)} labels:

{labels_text}

Classification:"""
    return prompt

def create_enhanced_text_with_improved_prompt(row, all_labels):
    question = str(row["QuestionText"]) if pd.notna(row["QuestionText"]) else ""
    mc_answer = str(row["MC_Answer"]) if pd.notna(row["MC_Answer"]) else ""
    explanation = str(row["StudentExplanation"]) if pd.notna(row["StudentExplanation"]) else ""
    return get_improved_compact_prompt(question, mc_answer, explanation, all_labels)

train_df["enhanced_text"] = train_df.apply(lambda row: create_enhanced_text_with_improved_prompt(row, all_labels), axis=1)

# ラベルエンコーディング
label_encoder = LabelEncoder()
train_df["encoded_labels"] = label_encoder.fit_transform(train_df["full_label"])

# データ分割
X_train, X_val, y_train, y_val = train_test_split(
    train_df["enhanced_text"].tolist(),
    train_df["encoded_labels"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=train_df["encoded_labels"].tolist(),
)

# データセットクラス
class ImprovedMathMisconceptionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# トークナイザー・モデル読み込み
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(MODEL_NAME)
config.num_labels = NUM_LABELS
config.problem_type = "single_label_classification"

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.pad_token_id = tokenizer.pad_token_id
model = prepare_model_for_kbit_training(model)
qlora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    modules_to_save=["classifier", "score"],
)
qlora_model = get_peft_model(model, qlora_config)

# データセット作成
train_dataset = ImprovedMathMisconceptionDataset(X_train, y_train, tokenizer, max_length=MAX_LENGTH)
val_dataset = ImprovedMathMisconceptionDataset(X_val, y_val, tokenizer, max_length=MAX_LENGTH)

# 評価指標
def compute_map3_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.softmax(torch.tensor(predictions), dim=-1).numpy()
    map_scores = []
    for i, true_label in enumerate(labels):
        top3_indices = np.argsort(predictions[i])[::-1][:3]
        score = 0.0
        for j, pred_idx in enumerate(top3_indices):
            if pred_idx == true_label:
                score = 1.0 / (j + 1)
                break
        map_scores.append(score)
    map3_score = np.mean(map_scores)
    accuracy = accuracy_score(labels, np.argmax(predictions, axis=1))
    return {"map3": map3_score, "accuracy": accuracy}

# 訓練設定
training_args = TrainingArguments(
    output_dir=f"./{NEW_MODEL_NAME}",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir=f"./logs/{NEW_MODEL_NAME}",
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="map3",
    greater_is_better=True,
    report_to="none",
    dataloader_pin_memory=False,
    fp16=False,
    bf16=True,
    optim="adamw_torch",
    learning_rate=1e-4,
    save_total_limit=3,
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    group_by_length=True,
)

# Trainer作成
trainer = Trainer(
    model=qlora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_map3_metrics,
)

# 訓練実行
trainer.train()

# モデル保存（実行ディレクトリに保存）
save_dir = f"./{NEW_MODEL_NAME}"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

# ラベルマッピング保存
import json
label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
with open(os.path.join(save_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, ensure_ascii=False, indent=2)

print(f"✅ モデル・トークナイザー・ラベルマッピングを {save_dir} に保存しました")