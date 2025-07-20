"""
MAP - DeBERTa-v3-xsmallモデルを使った6分類NLPモデル（コンペ形式準拠）

🎯 DeBERTa-v3-xsmall専用実装
正しいコンペ出力形式:
- True_Correct, True_Neither, True_Misconception
- False_Correct, False_Neither, False_Misconception

MC_Answer の正誤 × StudentExplanation の分類 = 6クラス分類

⚠️ 注意: DeBERTaモデルの読み込みに失敗した場合はプログラムを終了します
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")


class MathMisconceptionDataset(Dataset):
    """Math Misconception Dataset for PyTorch"""

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

        # トークン化
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


def load_and_prepare_data():
    """データの読み込みと前処理（6分類形式）"""
    print("=" * 60)
    print("データ読み込みと前処理（6分類形式）")
    print("=" * 60)

    # データ読み込み
    train_df = pd.read_csv("map_data/train.csv")
    test_df = pd.read_csv("map_data/test.csv")

    print(f"訓練データ: {train_df.shape}")
    print(f"テストデータ: {test_df.shape}")

    # コンペ形式のターゲット作成（6分類）
    # Category列が既に正しい形式になっている
    print("Category分布:")
    print(train_df["Category"].value_counts())

    # NaN値除去
    before_len = len(train_df)
    train_df = train_df.dropna(subset=["Category", "StudentExplanation"])
    after_len = len(train_df)
    print(f"NaN除去: {before_len} -> {after_len} ({before_len - after_len}行削除)")

    # テキスト特徴量作成
    def create_enhanced_text(row):
        """強化されたテキスト特徴量を作成"""
        question = str(row["QuestionText"]) if pd.notna(row["QuestionText"]) else ""
        mc_answer = str(row["MC_Answer"]) if pd.notna(row["MC_Answer"]) else ""
        explanation = (
            str(row["StudentExplanation"])
            if pd.notna(row["StudentExplanation"])
            else ""
        )

        # 質問、選択された答え、説明を結合
        enhanced_text = f"Question: {question} Selected Answer: {mc_answer} Explanation: {explanation}"
        return enhanced_text

    train_df["enhanced_text"] = train_df.apply(create_enhanced_text, axis=1)
    test_df["enhanced_text"] = test_df.apply(create_enhanced_text, axis=1)

    # 6つのカテゴリの確認
    unique_categories = sorted(train_df["Category"].unique())
    print(f"\nユニークなカテゴリ数: {len(unique_categories)}")
    print("カテゴリ一覧:")
    for i, cat in enumerate(unique_categories):
        count = (train_df["Category"] == cat).sum()
        print(f"  {i}: {cat} ({count}件)")

    return train_df, test_df


def prepare_model(num_labels=6, model_name="microsoft/deberta-v3-xsmall"):
    """モデルの準備（DeBERTa-v3-xsmall使用）"""
    print(f"\n" + "=" * 60)
    print(f"DeBERTaモデル準備: {model_name}")
    print("=" * 60)

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    try:
        # トークナイザーとモデルの読み込み
        print("DeBERTaトークナイザー読み込み中...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # DeBERTaはパディングトークンが設定されている
        print(f"パディングトークン: {tokenizer.pad_token}")

        # モデルの読み込み（6分類用）
        print("DeBERTaモデル読み込み中...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.float32,  # DeBERTa-xsmallは軽量なのでfloat32で十分
        )

        # デバイスに移動
        model = model.to(device)

        print(f"DeBERTaモデル読み込み完了: {num_labels}クラス分類")
        print(f"モデルパラメータ数: {model.num_parameters():,}")

        return model, tokenizer, device

    except Exception as e:
        print(f"❌ DeBERTaモデルの読み込みに失敗しました: {e}")
        print("\n考えられる原因:")
        print("1. インターネット接続の問題")
        print("2. transformersライブラリのバージョン問題")
        print("3. モデルのダウンロードに失敗")
        print("\n解決方法:")
        print("- インターネット接続確認")
        print("- pip install --upgrade transformers")
        print("- キャッシュクリア: rm -rf ~/.cache/huggingface/")
        print("\nプログラムを終了します。")
        exit(1)


def compute_map3_metrics(eval_pred):
    """MAP@3メトリクスの計算"""
    predictions, labels = eval_pred
    predictions = torch.softmax(torch.tensor(predictions), dim=-1).numpy()

    map_scores = []
    for i, true_label in enumerate(labels):
        # 上位3つの予測を取得
        top3_indices = np.argsort(predictions[i])[::-1][:3]

        # MAP計算
        score = 0.0
        for j, pred_idx in enumerate(top3_indices):
            if pred_idx == true_label:
                score = 1.0 / (j + 1)
                break
        map_scores.append(score)

    map3_score = np.mean(map_scores)
    accuracy = accuracy_score(labels, np.argmax(predictions, axis=1))

    return {"map3": map3_score, "accuracy": accuracy}


def fine_tune_model(train_df, model, tokenizer, device):
    """モデルのファインチューニング"""
    print(f"\n" + "=" * 60)
    print("6分類モデル ファインチューニング開始")
    print("=" * 60)

    # ラベルエンコーディング
    label_encoder = LabelEncoder()
    train_df["encoded_labels"] = label_encoder.fit_transform(train_df["Category"])

    print(f"エンコードされたラベル:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {i}: {label}")

    # データ分割
    train_texts = train_df["enhanced_text"].tolist()
    train_labels = train_df["encoded_labels"].tolist()

    # 少数クラス対策：stratifyを慎重に適用
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            train_texts,
            train_labels,
            test_size=0.2,
            random_state=42,
            stratify=train_labels,
        )
        print("Stratified split適用")
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            train_texts, train_labels, test_size=0.2, random_state=42
        )
        print("Regular split適用（少数クラスのため）")

    print(f"訓練データ: {len(X_train)}")
    print(f"検証データ: {len(X_val)}")

    # データセット作成（DeBERTa用）
    train_dataset = MathMisconceptionDataset(
        X_train, y_train, tokenizer, max_length=512  # DeBERTaの最大長
    )
    val_dataset = MathMisconceptionDataset(X_val, y_val, tokenizer, max_length=512)

    # データコレーター
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 訓練設定（DeBERTa-xsmall最適化）
    training_args = TrainingArguments(
        output_dir="./deberta_v3_xsmall_6class_model",
        num_train_epochs=5,  # DeBERTa-xsmallは軽量なので多めのエポック
        per_device_train_batch_size=16,  # xsmallは軽量なので大きなバッチサイズ
        per_device_eval_batch_size=16,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="steps",  # evaluation_strategy -> eval_strategy
        eval_steps=200,
        save_strategy="steps",
        save_steps=400,
        load_best_model_at_end=True,
        metric_for_best_model="map3",
        greater_is_better=True,
        report_to=None,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=2,  # 実効バッチサイズ=32
        fp16=device.type == "cuda",  # GPU使用時は半精度
        optim="adamw_torch",
        learning_rate=3e-5,  # DeBERTa推奨学習率
        save_total_limit=2,  # ストレージ節約
        remove_unused_columns=False,
    )

    # トレーナー設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_map3_metrics,
    )

    # ファインチューニング実行
    print("ファインチューニング開始...")
    trainer.train()

    # 最終評価
    print("\n最終評価:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")

    return trainer, label_encoder


def generate_submission(trainer, tokenizer, test_df, label_encoder):
    """コンペ形式の提出ファイル生成"""
    print(f"\n" + "=" * 60)
    print("コンペ形式提出ファイル生成")
    print("=" * 60)

    test_texts = test_df["enhanced_text"].tolist()
    test_dataset = MathMisconceptionDataset(
        test_texts, [0] * len(test_texts), tokenizer, max_length=512  # DeBERTa用
    )

    # 予測実行
    predictions = trainer.predict(test_dataset)

    # ソフトマックス適用
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

    # 各サンプルで上位3つの予測を取得
    submission_predictions = []
    for prob in probs:
        top3_indices = np.argsort(prob)[::-1][:3]
        top3_labels = [label_encoder.classes_[idx] for idx in top3_indices]
        submission_predictions.append(" ".join(top3_labels))

    # 提出形式のDataFrame作成
    submission_df = pd.DataFrame(
        {
            "row_id": range(len(test_df)),
            "Category:Misconception": submission_predictions,
        }
    )

    # サンプル表示
    print("提出ファイルサンプル:")
    print(submission_df.head(10))

    print(f"予測完了: {len(submission_df)}行")
    return submission_df


def main():
    """メイン実行関数"""
    print("🎯 MAP競技 - DeBERTa-v3-xsmall 6分類NLPモデル（コンペ形式準拠）")
    print("=" * 60)

    # データ準備
    train_df, test_df = load_and_prepare_data()

    # モデル準備（6分類）
    model, tokenizer, device = prepare_model(num_labels=6)

    # ファインチューニング
    trainer, label_encoder = fine_tune_model(train_df, model, tokenizer, device)

    # 提出ファイル生成
    submission_df = generate_submission(trainer, tokenizer, test_df, label_encoder)

    # 結果保存
    submission_df.to_csv("deberta_v3_xsmall_6class_submission.csv", index=False)
    print(f"\n提出ファイル保存: deberta_v3_xsmall_6class_submission.csv")

    print("\n" + "=" * 60)
    print("6分類モデル開発完了！")
    print("=" * 60)

    return trainer, label_encoder, submission_df


if __name__ == "__main__":
    print("🚀 MAP競技 - DeBERTa-v3-xsmall 6分類モデル開始")
    print("=" * 60)

    # システム要件チェック
    print("システム要件チェック:")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA利用可能: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    try:
        trainer, label_encoder, submission_df = main()
        print("\n🎉 DeBERTa-v3-xsmallモデル開発完了！")

    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによって中断されました")
        exit(0)

    except Exception as e:
        print(f"\n❌ 予期しないエラーが発生しました: {e}")
        print("\nDeBERTa-v3-xsmallモデルの実行には以下が必要です:")
        print("1. transformersライブラリ (pip install transformers)")
        print("2. 安定したインターネット接続（初回モデルダウンロード時）")
        print("3. 推奨メモリ: 4GB以上")
        print("\nプログラムを終了します。")
        exit(1)
