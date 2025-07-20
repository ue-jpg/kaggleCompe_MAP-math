"""
MAP競技 - データ確認とGemmaテストスクリプト

このスクリプトの目的:
1. ターゲット変数の分布確認
2. 実際の入力テキスト例の表示
3. Gemmaモデルの読み込みテスト
4. メモリ使用量の確認

実際の訓練前の事前確認用
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

warnings.filterwarnings("ignore")


def check_data_overview():
    """データ概要とターゲット変数の確認"""
    print("🔍 データ概要確認")
    print("=" * 60)

    try:
        # データ読み込み
        train_df = pd.read_csv("map_data/train.csv")
        test_df = pd.read_csv("map_data/test.csv")
        sample_submission = pd.read_csv("map_data/sample_submission.csv")

        print(f"✅ 訓練データ: {train_df.shape}")
        print(f"✅ テストデータ: {test_df.shape}")
        print(f"✅ サンプル提出: {sample_submission.shape}")

        # 基本統計
        print(f"\n📊 基本統計:")
        print(f"   総学生回答数: {len(train_df):,}")
        print(f"   ユニーク質問数: {train_df['QuestionId'].nunique()}")

        # カラム情報
        print(f"\n📋 カラム情報:")
        for col in train_df.columns:
            non_null = train_df[col].notna().sum()
            print(
                f"   {col}: {non_null:,}/{len(train_df):,} ({non_null/len(train_df)*100:.1f}%)"
            )

        return train_df, test_df, sample_submission

    except FileNotFoundError as e:
        print(f"❌ データファイルが見つかりません: {e}")
        print("map_data/フォルダに以下のファイルがあることを確認してください:")
        print("- train.csv")
        print("- test.csv")
        print("- sample_submission.csv")
        return None, None, None
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return None, None, None


def analyze_target_variables(train_df):
    """ターゲット変数の詳細分析"""
    if train_df is None:
        return

    print("\n🎯 ターゲット変数分析")
    print("=" * 60)

    # Category分析
    print("📊 Category分布:")
    category_dist = train_df["Category"].value_counts()
    for cat, count in category_dist.items():
        percentage = count / len(train_df) * 100
        print(f"   {cat}: {count:,}件 ({percentage:.1f}%)")

    # Misconception分析
    print(f"\n📊 Misconception分析:")
    misconception_dist = train_df["Misconception"].value_counts()
    print(f"   ユニークな誤概念数: {len(misconception_dist)}")
    print(f"   NA以外の誤概念: {(train_df['Misconception'] != 'NA').sum():,}件")

    print("\n   上位10の誤概念:")
    for i, (misc, count) in enumerate(misconception_dist.head(10).items(), 1):
        percentage = count / len(train_df) * 100
        print(f"   {i:2d}. {misc}: {count:,}件 ({percentage:.1f}%)")

    # 6分類の最終ターゲット
    print(f"\n🎯 6分類ターゲット（コンペ形式）:")
    print("   期待されるカテゴリ:")
    expected_categories = [
        "True_Correct",
        "True_Neither",
        "True_Misconception",
        "False_Correct",
        "False_Neither",
        "False_Misconception",
    ]

    actual_categories = sorted(train_df["Category"].unique())
    print(f"   実際のカテゴリ数: {len(actual_categories)}")

    for i, cat in enumerate(actual_categories):
        count = (train_df["Category"] == cat).sum()
        percentage = count / len(train_df) * 100
        status = "✅" if cat in expected_categories else "⚠️"
        print(f"   {i}: {status} {cat}: {count:,}件 ({percentage:.1f}%)")


def show_text_examples(train_df, test_df):
    """実際の入力テキスト例を表示"""
    if train_df is None or test_df is None:
        return

    print("\n📝 入力テキスト例")
    print("=" * 60)

    def create_enhanced_text(row):
        """強化されたテキスト特徴量を作成"""
        question = str(row["QuestionText"]) if pd.notna(row["QuestionText"]) else ""
        mc_answer = str(row["MC_Answer"]) if pd.notna(row["MC_Answer"]) else ""
        explanation = (
            str(row["StudentExplanation"])
            if pd.notna(row["StudentExplanation"])
            else ""
        )

        enhanced_text = f"Question: {question} Selected Answer: {mc_answer} Explanation: {explanation}"
        return enhanced_text

    # 各カテゴリから1例ずつ表示
    categories = train_df["Category"].unique()

    for i, category in enumerate(categories[:3]):  # 最初の3カテゴリ
        print(f"\n📋 例 {i+1}: カテゴリ = {category}")
        print("-" * 40)

        sample = train_df[train_df["Category"] == category].iloc[0]
        enhanced_text = create_enhanced_text(sample)

        print(f"   質問ID: {sample['QuestionId']}")
        print(f"   選択答え: {sample['MC_Answer']}")
        print(f"   カテゴリ: {sample['Category']}")
        print(f"   誤概念: {sample['Misconception']}")
        print(f"\n   📄 完全な入力テキスト:")
        print(f"   {enhanced_text}")
        print(f"\n   📏 テキスト長: {len(enhanced_text)}文字")

    # テキスト長統計
    train_df["enhanced_text"] = train_df.apply(create_enhanced_text, axis=1)
    text_lengths = train_df["enhanced_text"].str.len()

    print(f"\n📊 テキスト長統計:")
    print(f"   平均: {text_lengths.mean():.1f}文字")
    print(f"   中央値: {text_lengths.median():.1f}文字")
    print(f"   最小: {text_lengths.min()}文字")
    print(f"   最大: {text_lengths.max()}文字")
    print(
        f"   512文字超過: {(text_lengths > 512).sum():,}件 ({(text_lengths > 512).mean()*100:.1f}%)"
    )


def test_gemma_loading():
    """Gemmaモデルの読み込みテスト"""
    print("\n🤖 Gemmaモデル読み込みテスト")
    print("=" * 60)

    # システム情報
    print("💻 システム情報:")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA利用可能: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"   GPU総メモリ: {total_memory / 1e9:.1f} GB")

        # メモリ使用量チェック
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0)
        cached = torch.cuda.memory_reserved(0)
        print(f"   GPU使用メモリ: {allocated / 1e9:.2f} GB")
        print(f"   GPUキャッシュ: {cached / 1e9:.2f} GB")

    # Gemmaトークナイザーテスト
    print(f"\n🔤 Gemmaトークナイザーテスト:")
    try:
        print("   トークナイザー読み込み中...")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("   ✅ トークナイザー読み込み成功")
        print(f"   語彙サイズ: {len(tokenizer):,}")

        # テストトークン化
        test_text = "Question: What is 2+2? Selected Answer: 4 Explanation: I added 2 and 2 to get 4."
        tokens = tokenizer(
            test_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        print(f"   テストトークン化: {tokens['input_ids'].shape[1]}トークン")

        return True, tokenizer

    except Exception as e:
        print(f"   ❌ トークナイザー読み込み失敗: {e}")
        return False, None


def test_gemma_model(tokenizer_success, tokenizer):
    """Gemmaモデル本体のテスト"""
    if not tokenizer_success:
        print("\n❌ トークナイザー失敗のためモデルテストをスキップ")
        return False

    print(f"\n🧠 Gemmaモデル本体テスト:")

    try:
        print("   モデル読み込み中...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "google/gemma-2b",
            num_labels=6,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )

        print("   ✅ モデル読み込み成功")
        print(f"   パラメータ数: {model.num_parameters():,}")

        # メモリ使用量（GPU使用時）
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            print(f"   モデル読み込み後GPU使用量: {allocated / 1e9:.2f} GB")

        # 簡単な推論テスト
        print("   推論テスト実行中...")
        test_text = "Question: What is 2+2? Selected Answer: 4 Explanation: I added 2 and 2 to get 4."
        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)

        print(f"   ✅ 推論テスト成功")
        print(f"   出力形状: {predictions.shape}")
        print(f"   予測確率例: {predictions[0][:3].tolist()}")

        # メモリクリーンアップ
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"   ❌ モデル読み込み失敗: {e}")
        print(f"\n💡 解決策:")
        print(f"   1. Hugging Face認証: huggingface-cli login")
        print(f"   2. メモリ確保: より小さなバッチサイズ使用")
        print(f"   3. インターネット接続確認")
        return False


def main():
    """メイン実行関数"""
    print("🔍 MAP競技 - データ確認とGemmaテストスクリプト")
    print("=" * 60)
    print("実際の訓練前の事前確認を行います\n")

    # 1. データ確認
    train_df, test_df, sample_submission = check_data_overview()

    if train_df is not None:
        # 2. ターゲット変数分析
        analyze_target_variables(train_df)

        # 3. テキスト例表示
        show_text_examples(train_df, test_df)

    # 4. Gemmaテスト
    tokenizer_success, tokenizer = test_gemma_loading()

    if tokenizer_success:
        model_success = test_gemma_model(tokenizer_success, tokenizer)
    else:
        model_success = False

    # 5. 最終結果
    print(f"\n🏁 事前確認結果")
    print("=" * 60)
    print(f"   データ読み込み: {'✅' if train_df is not None else '❌'}")
    print(f"   Gemmaトークナイザー: {'✅' if tokenizer_success else '❌'}")
    print(f"   Gemmaモデル: {'✅' if model_success else '❌'}")

    if train_df is not None and tokenizer_success and model_success:
        print(f"\n🎉 全ての事前確認が完了しました！")
        print(f"   map_6class_model.py の実行準備が整いました。")
    else:
        print(f"\n⚠️  いくつかの問題があります。解決してから本実行してください。")

    return train_df is not None, tokenizer_success, model_success


if __name__ == "__main__":
    data_ok, tokenizer_ok, model_ok = main()
