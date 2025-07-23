#!/usr/bin/env python
"""
改良版QLoRAモデルをKaggleオフライン環境用に統合するスクリプト
QLoRAアダプタをベースモデルにマージして単一のモデルファイルを作成
"""
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


def merge_improved_qlora_model_for_kaggle():
    """改良版QLoRAモデルを統合してKaggle用モデルを作成"""
    print("🚀 改良版QLoRA→Kaggle用統合モデル作成")
    print("=" * 60)

    # パス設定
    IMPROVED_MODEL_PATH = r"C:\Users\mouse\Desktop\NotDelete\GitHub\kaggleCompe_MAP-math\colab\colabで訓練して保存\gemma-2-2b-improved-prompts-qlora"
    OUTPUT_PATH = r"C:\Users\mouse\Desktop\NotDelete\GitHub\kaggleCompe_MAP-math\colab\colabで訓練して保存\kaggle-ready-improved-qlora"

    try:
        # 改良版ラベルマッピング読み込み
        improved_label_file = os.path.join(
            IMPROVED_MODEL_PATH, "improved_label_mapping.json"
        )
        if not os.path.exists(improved_label_file):
            print(f"❌ 改良版ラベルマッピングが見つかりません: {improved_label_file}")
            return None

        with open(improved_label_file, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
        print(f"✅ 改良版ラベル数: {len(label_mapping)}")

        # 改良版メタデータ読み込み
        metadata_file = os.path.join(
            IMPROVED_MODEL_PATH, "improved_model_metadata.json"
        )
        if os.path.exists(metadata_file):
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            print(f"📋 改良版メタデータ読み込み完了")
            print(f"  🔸 モデル名: {metadata.get('improved_model_name', 'Unknown')}")
            print(
                f"  🔸 QLoRA適用: {metadata.get('improvements', {}).get('qlora_applied', False)}"
            )
            print(
                f"  🔸 プロンプト最適化: {metadata.get('improvements', {}).get('prompt_optimization', 'Unknown')}"
            )
        else:
            metadata = {}
            print("⚠️ メタデータファイルが見つかりません")

        # アダプター設定読み込み
        adapter_config_path = os.path.join(IMPROVED_MODEL_PATH, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            print(f"❌ アダプター設定が見つかりません: {adapter_config_path}")
            return None

        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)

        base_model_name = adapter_config.get(
            "base_model_name_or_path", "google/gemma-2-2b-it"
        )
        print(f"📦 ベースモデル: {base_model_name}")
        print(f"🔧 QLoRA設定:")
        print(f"  🔸 r (rank): {adapter_config.get('r', 'Unknown')}")
        print(f"  🔸 lora_alpha: {adapter_config.get('lora_alpha', 'Unknown')}")
        print(f"  🔸 target_modules: {adapter_config.get('target_modules', [])}")

        # ベースモデル読み込み
        print("\n🧠 ベースモデル読み込み中...")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(label_mapping),
            torch_dtype=torch.float32,  # QLoRA統合のためfloat32使用
            trust_remote_code=True,
        )
        print(
            f"✅ ベースモデル読み込み完了 (パラメータ数: {sum(p.numel() for p in base_model.parameters()):,})"
        )

        # QLoRA PEFTアダプター適用
        print("⚡ QLoRA PEFTアダプター読み込み中...")
        try:
            peft_model = PeftModel.from_pretrained(base_model, IMPROVED_MODEL_PATH)
            print("✅ QLoRAアダプター読み込み完了")
        except Exception as e:
            print(f"❌ PEFTアダプター読み込み失敗: {e}")
            return None

        # アダプターをベースモデルにマージ
        print("🔀 QLoRAアダプターをベースモデルにマージ中...")
        try:
            merged_model = peft_model.merge_and_unload()
            print("✅ マージ完了 - 改良版統合モデル作成成功")
        except Exception as e:
            print(f"❌ マージ失敗: {e}")
            return None

        # トークナイザー読み込み
        print("📝 改良版トークナイザー読み込み中...")
        tokenizer = AutoTokenizer.from_pretrained(IMPROVED_MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ トークナイザー読み込み完了")

        # 出力ディレクトリ作成
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        print(f"📁 出力ディレクトリ作成: {OUTPUT_PATH}")

        # 統合モデル保存
        print(f"💾 改良版統合モデル保存中...")
        merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
        tokenizer.save_pretrained(OUTPUT_PATH)
        print("✅ モデルとトークナイザー保存完了")

        # 改良版ラベルマッピングコピー
        import shutil

        shutil.copy2(
            improved_label_file, os.path.join(OUTPUT_PATH, "label_mapping.json")
        )
        print("✅ 改良版ラベルマッピングコピー完了")

        # Kaggle用改良版モデル情報保存
        kaggle_model_info = {
            "model_type": "merged_improved_qlora_model",
            "base_model": base_model_name,
            "peft_type": "QLoRA",
            "num_labels": len(label_mapping),
            "created_for": "kaggle_offline_environment",
            "usage": "Use AutoModelForSequenceClassification.from_pretrained() directly",
            "improvements": {
                "prompt_optimization": metadata.get("improvements", {}).get(
                    "prompt_optimization", "final_compact_prompt.py based"
                ),
                "qlora_applied": True,
                "quantization": "4-bit nf4 (merged to float32)",
                "label_coverage": f"{len(label_mapping)} labels including False_Correct:NA",
                "prompt_features": metadata.get("improvements", {}).get(
                    "prompt_features", []
                ),
            },
            "original_qlora_config": {
                "r": adapter_config.get("r"),
                "lora_alpha": adapter_config.get("lora_alpha"),
                "target_modules": adapter_config.get("target_modules"),
                "task_type": adapter_config.get("task_type"),
            },
            "training_info": metadata.get("training_info", {}),
            "created_from": "gemma-2-2b-improved-prompts-qlora",
        }

        kaggle_info_path = os.path.join(OUTPUT_PATH, "kaggle_model_info.json")
        with open(kaggle_info_path, "w", encoding="utf-8") as f:
            json.dump(kaggle_model_info, f, indent=2, ensure_ascii=False)
        print("✅ Kaggle用改良版モデル情報保存完了")

        # ファイルサイズ確認
        print(f"\n📦 作成された改良版統合モデルファイル:")
        total_size = 0
        for file in os.listdir(OUTPUT_PATH):
            file_path = os.path.join(OUTPUT_PATH, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                total_size += size_mb
                print(f"  📄 {file}: {size_mb:.1f} MB")
        print(f"📊 総サイズ: {total_size:.1f} MB")

        # 使用方法説明
        print(f"\n🎯 Kaggleでの改良版モデル使用方法:")
        print(f"1. {OUTPUT_PATH} フォルダをKaggleデータセットとしてアップロード")
        print(f"2. データセット名: 'gemma-2b-improved-prompts-kaggle' などに設定")
        print(f"3. ノートブックでの読み込み:")
        print(f"   ```python")
        print(
            f"   from transformers import AutoModelForSequenceClassification, AutoTokenizer"
        )
        print(f"   import json")
        print(f"   ")
        print(f"   # モデル読み込み")
        print(
            f"   model = AutoModelForSequenceClassification.from_pretrained('/kaggle/input/your-dataset-name')"
        )
        print(
            f"   tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/your-dataset-name')"
        )
        print(f"   ")
        print(f"   # ラベルマッピング読み込み")
        print(
            f"   with open('/kaggle/input/your-dataset-name/label_mapping.json', 'r') as f:"
        )
        print(f"       label_mapping = json.load(f)")
        print(f"   ```")
        print(f"4. 推論時は改良プロンプト形式を使用してください")

        print(f"\n✨ 改良版QLoRA→Kaggle統合モデル作成完了!")
        print(f"🎉 プロンプト最適化 + QLoRA効果がKaggleで利用可能になりました")

        return OUTPUT_PATH

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = merge_improved_qlora_model_for_kaggle()
    if result:
        print(f"\n🚀 成功: {result}")
    else:
        print(f"\n💔 失敗: モデル統合に失敗しました")
