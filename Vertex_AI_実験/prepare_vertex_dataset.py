#!/usr/bin/env python3
"""
VertexAI用データセット準備スクリプト

MAP競技の訓練データを、VertexAIでのファインチューニングに適した形式に変換します。
- テキスト: Question + Selected Answer + Student Explanation
- ラベル: 6つのカテゴリ (True/False_Correct/Neither/Misconception)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def create_enhanced_text(row):
    """質問、選択答え、学生説明を統合したテキストを作成"""
    question = str(row["QuestionText"]) if pd.notna(row["QuestionText"]) else ""
    mc_answer = str(row["MC_Answer"]) if pd.notna(row["MC_Answer"]) else ""
    explanation = (
        str(row["StudentExplanation"]) if pd.notna(row["StudentExplanation"]) else ""
    )

    # 統合テキスト作成
    enhanced_text = f"Question: {question} Selected Answer: {mc_answer} Student Explanation: {explanation}"
    return enhanced_text.strip()


def create_category_label(row):
    """Categoryカラムから6つの基本カテゴリを作成"""
    category = str(row["Category"])

    # 6つの基本カテゴリにマッピング
    if category == "True_Correct":
        return "True_Correct"
    elif category == "True_Neither":
        return "True_Neither"
    elif category == "True_Misconception":
        return "True_Misconception"
    elif category == "False_Correct":
        return "False_Correct"
    elif category == "False_Neither":
        return "False_Neither"
    elif category == "False_Misconception":
        return "False_Misconception"
    else:
        return "Unknown"


def prepare_vertex_dataset():
    """VertexAI用のデータセットを準備"""

    print("🚀 VertexAI用データセット準備開始")

    # データ読み込み
    train_path = "../map_data/train.csv"
    if not os.path.exists(train_path):
        print(f"❌ エラー: {train_path} が見つかりません")
        return False

    print(f"📊 データ読み込み: {train_path}")
    df = pd.read_csv(train_path)
    print(f"   データ形状: {df.shape}")

    # データの基本情報表示
    print("\n📋 カラム情報:")
    for col in df.columns:
        print(f"   {col}: {df[col].dtype}, 欠損値: {df[col].isnull().sum()}")

    # 強化テキスト作成
    print("\n🔧 強化テキスト作成中...")
    df["enhanced_text"] = df.apply(create_enhanced_text, axis=1)

    # カテゴリラベル作成
    print("🏷️ カテゴリラベル作成中...")
    df["category_label"] = df.apply(create_category_label, axis=1)

    # Unknown カテゴリの確認
    unknown_count = (df["category_label"] == "Unknown").sum()
    if unknown_count > 0:
        print(f"⚠️ 警告: Unknown カテゴリが {unknown_count} 件あります")
        print("Unknown カテゴリの元データ:")
        print(df[df["category_label"] == "Unknown"]["Category"].value_counts())

    # Unknownを除外
    df_clean = df[df["category_label"] != "Unknown"].copy()
    print(f"📊 クリーン後のデータ形状: {df_clean.shape}")

    # カテゴリ分布確認
    print("\n📊 カテゴリ分布:")
    category_counts = df_clean["category_label"].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(df_clean)) * 100
        print(f"   {category}: {count:,} ({percentage:.1f}%)")

    # VertexAI用データセット作成
    vertex_dataset = df_clean[["enhanced_text", "category_label"]].copy()
    vertex_dataset.columns = ["text", "label"]

    # テキスト長の統計
    text_lengths = vertex_dataset["text"].str.len()
    print(f"\n📝 テキスト長統計:")
    print(f"   平均: {text_lengths.mean():.1f} 文字")
    print(f"   中央値: {text_lengths.median():.1f} 文字")
    print(f"   最大: {text_lengths.max()} 文字")
    print(f"   最小: {text_lengths.min()} 文字")
    print(
        f"   512文字以下: {(text_lengths <= 512).sum()} ({(text_lengths <= 512).mean()*100:.1f}%)"
    )

    # データセット保存
    output_path = "vertex_training_dataset.csv"
    vertex_dataset.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n💾 データセット保存完了: {output_path}")
    print(f"   保存データ形状: {vertex_dataset.shape}")

    # サンプルデータ表示
    print("\n📋 サンプルデータ:")
    for i, (_, row) in enumerate(vertex_dataset.head(3).iterrows()):
        print(f"\n--- サンプル {i+1} ---")
        print(f"ラベル: {row['label']}")
        print(f"テキスト: {row['text'][:200]}...")

    # 訓練/検証分割用のデータセットも作成
    from sklearn.model_selection import train_test_split

    print("\n🔄 訓練/検証分割データセット作成中...")
    train_data, val_data = train_test_split(
        vertex_dataset, test_size=0.2, random_state=42, stratify=vertex_dataset["label"]
    )

    # 分割データ保存
    train_data.to_csv("vertex_train_dataset.csv", index=False, encoding="utf-8")
    val_data.to_csv("vertex_val_dataset.csv", index=False, encoding="utf-8")

    print(f"   訓練データ: {train_data.shape} → vertex_train_dataset.csv")
    print(f"   検証データ: {val_data.shape} → vertex_val_dataset.csv")

    print("\n✅ VertexAI用データセット準備完了！")

    return True


def show_dataset_info():
    """作成されたデータセットの情報を表示"""
    files = [
        "vertex_training_dataset.csv",
        "vertex_train_dataset.csv",
        "vertex_val_dataset.csv",
    ]

    print("\n📂 作成されたファイル:")
    for file in files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            print(f"   {file}: {df.shape}")
            print(f"      ラベル数: {df['label'].nunique()}")
            print(f"      サイズ: {os.path.getsize(file) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    # フォルダ確認・作成
    vertex_folder = Path(
        "C:/Users/mouse/Desktop/P/learnMachineLearning/kaggleはじめてのコンペ/Vertex_AI_実験"
    )
    if not vertex_folder.exists():
        vertex_folder.mkdir(parents=True, exist_ok=True)
        print(f"📁 フォルダ作成: {vertex_folder}")

    # フォルダに移動
    os.chdir(vertex_folder)
    print(f"📍 作業ディレクトリ: {os.getcwd()}")

    # データセット準備実行
    success = prepare_vertex_dataset()

    if success:
        show_dataset_info()
        print("\n🎉 すべて完了！VertexAIでの実験準備が整いました。")
    else:
        print("\n❌ データセット準備に失敗しました。")
