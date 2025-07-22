#!/usr/bin/env pytho# 設定
MODEL_PATH = "./colab/colabで訓練して保存/kaggle-ready-model"  # 統合モデルのパス
DATA_PATH = "./map_data/train.csv"  # 訓練データのパス
# -*- coding: utf-8 -*-
"""
Kaggle Ready Model Performance Evaluation
統合モデル（kaggle-ready-model）の性能評価スクリプト

train.csvのデータを使用して予測を行い、実際のラベルと比較します。
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# 設定
MODEL_PATH = r"c:\Users\mouse\Desktop\NotDelete\GitHub\kaggleCompe_MAP-math\colab\colabで訓練して保存\kaggle-ready-model"  # 統合モデルのパス
DATA_PATH = r"c:\Users\mouse\Desktop\NotDelete\GitHub\kaggleCompe_MAP-math\map_data\train.csv"  # 訓練データのパス
MAX_LENGTH = 512
BATCH_SIZE = 2  # バッチサイズ（メモリ不足対策で小さく）

# カテゴリマッピング（6分類）
CATEGORY_MAPPING = {
    0: "True",
    1: "False",
    2: "Correct",
    3: "Neither",
    4: "Misconception",
    5: "Unknown",  # 予備（実際のデータに依存）
}


def create_enhanced_text(row):
    """Question + MC_Answer + Student Explanation を統合"""
    question = str(row["QuestionText"]) if pd.notna(row["QuestionText"]) else ""
    mc_answer = str(row["MC_Answer"]) if pd.notna(row["MC_Answer"]) else ""
    explanation = (
        str(row["StudentExplanation"]) if pd.notna(row["StudentExplanation"]) else ""
    )

    enhanced_text = (
        f"Question: {question} Selected Answer: {mc_answer} Explanation: {explanation}"
    )
    return enhanced_text


def load_model_and_tokenizer(model_path):
    """統合モデルとトークナイザーの読み込み"""
    print("=" * 60)
    print("🤖 統合モデル読み込み")
    print("=" * 60)

    try:
        # モデルパスの確認
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"モデルディレクトリが見つかりません: {model_path}")

        print(f"📁 モデルパス: {model_path}")

        # トークナイザー読み込み
        print("📝 トークナイザー読み込み中...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        # パディングトークンの設定
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"✅ トークナイザー読み込み完了")
        print(f"🔖 パディングトークン: {tokenizer.pad_token}")

        # モデル読み込み
        print("🧠 モデル読み込み中（CPU専用、低メモリモード）...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,  # CPU用
            low_cpu_mem_usage=True,  # 低メモリ使用モード
        )

        # デバイス設定（CPUのみ）
        device = torch.device("cpu")
        model.to(device)
        model.eval()

        print(f"✅ モデル読み込み完了")
        print(f"🖥️ デバイス: {device}")
        print(f"📊 分類クラス数: {model.config.num_labels}")

        return model, tokenizer, device

    except Exception as e:
        print(f"❌ モデル読み込み失敗: {e}")
        return None, None, None


def load_and_prepare_data(data_path):
    """データの読み込みと前処理"""
    print("\n" + "=" * 60)
    print("📊 データ読み込みと前処理")
    print("=" * 60)

    try:
        # データ読み込み
        df = pd.read_csv(data_path)
        print(f"✅ データ読み込み: {df.shape}")

        # NaN値の除去
        before_len = len(df)
        df = df.dropna(subset=["Category", "StudentExplanation"])
        after_len = len(df)
        print(
            f"🧹 NaN除去: {before_len} -> {after_len} ({before_len - after_len}行削除)"
        )

        # カテゴリ分布の確認
        print("\n📋 Category分布:")
        category_counts = df["Category"].value_counts()
        print(category_counts)

        # ラベルエンコーディング
        unique_categories = sorted(df["Category"].unique())
        category_to_id = {cat: i for i, cat in enumerate(unique_categories)}
        id_to_category = {i: cat for cat, i in category_to_id.items()}

        df["label_id"] = df["Category"].map(category_to_id)

        print(f"\n📊 ラベルマッピング:")
        for label_id, category in id_to_category.items():
            count = (df["label_id"] == label_id).sum()
            print(f"  {label_id}: {category} ({count:,}件)")

        # 強化テキスト特徴量作成
        print("\n🔧 強化テキスト特徴量作成中...")
        df["enhanced_text"] = df.apply(create_enhanced_text, axis=1)

        # テキスト長統計
        text_lengths = df["enhanced_text"].str.len()
        print(f"\n📝 テキスト長統計:")
        print(f"  平均: {text_lengths.mean():.1f} 文字")
        print(f"  中央値: {text_lengths.median():.1f} 文字")
        print(f"  最大: {text_lengths.max()} 文字")
        print(
            f"  512文字以下: {(text_lengths <= 512).sum()} ({(text_lengths <= 512).mean()*100:.1f}%)"
        )

        return df, id_to_category

    except Exception as e:
        print(f"❌ データ読み込み失敗: {e}")
        return None, None


def predict_batch(model, tokenizer, texts, device, max_length=512):
    """バッチ予測"""
    predictions = []
    probabilities = []

    # トークン化
    encoded = tokenizer(
        texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )

    # デバイスに移動
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # 予測
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        probabilities.extend(probs.cpu().numpy())

    return predictions, probabilities


def evaluate_model(model, tokenizer, df, id_to_category, device):
    """モデルの評価"""
    print("\n" + "=" * 60)
    print("🎯 モデル評価実行")
    print("=" * 60)

    texts = df["enhanced_text"].tolist()
    true_labels = df["label_id"].tolist()

    all_predictions = []
    all_probabilities = []

    # バッチごとに予測
    print(f"📊 総サンプル数: {len(texts)}")
    print(f"🔄 バッチサイズ: {BATCH_SIZE}")

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_predictions, batch_probabilities = predict_batch(
            model, tokenizer, batch_texts, device, MAX_LENGTH
        )

        all_predictions.extend(batch_predictions)
        all_probabilities.extend(batch_probabilities)

        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"  処理済み: {min(i + BATCH_SIZE, len(texts)):,} / {len(texts):,}")

    print("✅ 予測完了")

    # 精度計算
    accuracy = accuracy_score(true_labels, all_predictions)
    print(f"\n📈 全体精度: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # 分類レポート
    print("\n📊 分類レポート:")
    target_names = [id_to_category[i] for i in sorted(id_to_category.keys())]

    # 実際に存在するラベルのみを取得
    unique_labels = sorted(list(set(true_labels)))
    labels_present = [i for i in sorted(id_to_category.keys()) if i in unique_labels]
    target_names_present = [id_to_category[i] for i in labels_present]

    print(
        classification_report(
            true_labels,
            all_predictions,
            labels=labels_present,
            target_names=target_names_present,
            digits=4,
        )
    )

    # 混同行列
    print("\n📋 混同行列:")
    cm = confusion_matrix(true_labels, all_predictions, labels=labels_present)
    cm_df = pd.DataFrame(cm, index=target_names_present, columns=target_names_present)
    print(cm_df)

    return all_predictions, all_probabilities


def create_detailed_results(df, predictions, probabilities, id_to_category):
    """詳細な結果の作成"""
    print("\n" + "=" * 60)
    print("📋 詳細結果作成")
    print("=" * 60)

    # 結果データフレーム作成
    results_df = df.copy()
    results_df["predicted_label_id"] = predictions
    results_df["predicted_category"] = [id_to_category[pred] for pred in predictions]
    results_df["is_correct"] = (
        results_df["label_id"] == results_df["predicted_label_id"]
    )

    # 予測確率の追加
    prob_array = np.array(probabilities)
    for i, category in id_to_category.items():
        results_df[f"prob_{category}"] = prob_array[:, i]

    # 最大確率の追加
    results_df["max_probability"] = prob_array.max(axis=1)

    # 結果統計
    correct_count = results_df["is_correct"].sum()
    total_count = len(results_df)
    accuracy = correct_count / total_count

    print(f"✅ 詳細結果作成完了")
    print(f"📊 正解数: {correct_count:,} / {total_count:,} ({accuracy*100:.2f}%)")

    return results_df


def save_results(results_df, output_path="model_evaluation_results.csv"):
    """結果の保存"""
    print(f"\n💾 結果保存: {output_path}")

    try:
        # 主要な列を選択して保存
        columns_to_save = [
            "QuestionId",
            "QuestionText",
            "MC_Answer",
            "StudentExplanation",
            "Category",
            "predicted_category",
            "is_correct",
            "max_probability",
        ]

        # 確率列も追加
        prob_columns = [col for col in results_df.columns if col.startswith("prob_")]
        columns_to_save.extend(prob_columns)

        # 存在する列のみを選択
        available_columns = [
            col for col in columns_to_save if col in results_df.columns
        ]

        save_df = results_df[available_columns]
        save_df.to_csv(output_path, index=False, encoding="utf-8")

        print(f"✅ 結果保存完了: {len(save_df)} 行, {len(available_columns)} 列")
        print(f"📁 保存ファイル: {output_path}")

        return True

    except Exception as e:
        print(f"❌ 結果保存失敗: {e}")
        return False


def show_sample_predictions(results_df, num_samples=10):
    """サンプル予測結果の表示"""
    print("\n" + "=" * 60)
    print("🔍 サンプル予測結果")
    print("=" * 60)

    # 正解と不正解のサンプルを表示
    correct_samples = results_df[results_df["is_correct"] == True].head(
        num_samples // 2
    )
    incorrect_samples = results_df[results_df["is_correct"] == False].head(
        num_samples // 2
    )

    print("✅ 正解サンプル:")
    for idx, (_, row) in enumerate(correct_samples.iterrows()):
        print(f"\n[{idx+1}] QuestionId: {row.get('QuestionId', 'N/A')}")
        print(f"  実際: {row['Category']} → 予測: {row['predicted_category']}")
        print(f"  確率: {row['max_probability']:.3f}")
        print(f"  学生説明: {row['StudentExplanation'][:100]}...")

    print("\n❌ 不正解サンプル:")
    for idx, (_, row) in enumerate(incorrect_samples.iterrows()):
        print(f"\n[{idx+1}] QuestionId: {row.get('QuestionId', 'N/A')}")
        print(f"  実際: {row['Category']} → 予測: {row['predicted_category']}")
        print(f"  確率: {row['max_probability']:.3f}")
        print(f"  学生説明: {row['StudentExplanation'][:100]}...")


def main():
    """メイン処理"""
    print("🚀 Kaggle Ready Model 性能評価開始")
    print("=" * 80)

    # モデル読み込み
    model, tokenizer, device = load_model_and_tokenizer(MODEL_PATH)
    if model is None:
        print("❌ モデル読み込みに失敗しました")
        return

    # データ読み込み
    df, id_to_category = load_and_prepare_data(DATA_PATH)
    if df is None:
        print("❌ データ読み込みに失敗しました")
        return

    # サンプル数制限（メモリ節約のため）
    sample_size = min(100, len(df))  # 最大100サンプル（メモリ不足対策）
    if len(df) > sample_size:
        print(f"\n⚠️ サンプル数を制限: {len(df)} → {sample_size}")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # モデル評価
    predictions, probabilities = evaluate_model(
        model, tokenizer, df, id_to_category, device
    )

    # 詳細結果作成
    results_df = create_detailed_results(df, predictions, probabilities, id_to_category)

    # 結果保存
    save_results(results_df)

    # サンプル表示
    show_sample_predictions(results_df)

    print("\n🎉 評価完了!")
    print("📁 詳細結果は 'model_evaluation_results.csv' に保存されました")


if __name__ == "__main__":
    main()
