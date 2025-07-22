#!/usr/bin/env python
"""
PEFTモデルをKaggleオフライン環境用に統合するスクリプト
LoRAアダプタをベースモデルにマージして単一のモデルファイルを作成
"""
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


def merge_peft_model_for_kaggle():
    """PEFTモデルを統合してKaggle用モデルを作成"""
    print("🔧 Kaggle用統合モデル作成")
    print("=" * 50)

    # パス設定
    MODEL_DATA_PATH = r"C:\Users\mouse\Desktop\NotDelete\GitHub\kaggleCompe_MAP-math\colab\colabで訓練して保存\gemma-2-2b-math-model"
    OUTPUT_PATH = r"C:\Users\mouse\Desktop\NotDelete\GitHub\kaggleCompe_MAP-math\colab\colabで訓練して保存\kaggle-ready-model"

    try:
        # ラベルマッピング読み込み
        label_file = os.path.join(MODEL_DATA_PATH, "label_mapping.json")
        with open(label_file, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
        print(f"✅ ラベル数: {len(label_mapping)}")

        # アダプター設定読み込み
        adapter_config_path = os.path.join(MODEL_DATA_PATH, "adapter_config.json")
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)

        base_model_name = adapter_config.get(
            "base_model_name_or_path", "google/gemma-2-2b-it"
        )
        print(f"📦 ベースモデル: {base_model_name}")

        # ベースモデル読み込み
        print("🧠 ベースモデル読み込み中...")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(label_mapping),
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        # PEFTアダプター適用
        print("🔧 PEFTアダプター読み込み中...")
        peft_model = PeftModel.from_pretrained(base_model, MODEL_DATA_PATH)

        # アダプターをベースモデルにマージ
        print("🔀 アダプターをベースモデルにマージ中...")
        merged_model = peft_model.merge_and_unload()

        # トークナイザー読み込み
        print("📝 トークナイザー読み込み中...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DATA_PATH)

        # 出力ディレクトリ作成
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        # 統合モデル保存
        print(f"💾 統合モデル保存中: {OUTPUT_PATH}")
        merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
        tokenizer.save_pretrained(OUTPUT_PATH)

        # ラベルマッピングコピー
        import shutil

        shutil.copy2(label_file, os.path.join(OUTPUT_PATH, "label_mapping.json"))

        # モデル情報保存
        model_info = {
            "model_type": "merged_peft_model",
            "base_model": base_model_name,
            "peft_type": adapter_config.get("peft_type"),
            "num_labels": len(label_mapping),
            "created_for": "kaggle_offline_environment",
            "usage": "Use AutoModelForSequenceClassification.from_pretrained() directly",
        }

        with open(os.path.join(OUTPUT_PATH, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)

        print("✅ 統合モデル作成完了!")
        print(f"📁 出力パス: {OUTPUT_PATH}")
        print("\n📦 作成されたファイル:")
        for file in os.listdir(OUTPUT_PATH):
            file_path = os.path.join(OUTPUT_PATH, file)
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  {file}: {size:.1f} MB")

        print(f"\n🎯 Kaggleでの使用方法:")
        print(f"1. {OUTPUT_PATH} フォルダをKaggleデータセットとしてアップロード")
        print(f"2. ノートブックでMODEL_DATA_PATHを統合モデルのパスに設定")
        print(f"3. AutoModelForSequenceClassification.from_pretrained()で直接読み込み")

        return OUTPUT_PATH

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    merge_peft_model_for_kaggle()
