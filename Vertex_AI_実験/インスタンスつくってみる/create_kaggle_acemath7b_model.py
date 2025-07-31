import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


def merge_acemath7b_qlora_for_kaggle():
    print("🚀 AceMath-7B QLoRAモデルをKaggle用に統合します")
    print("=" * 60)

    # パス設定
    IMPROVED_MODEL_PATH = r"C:\Users\mouse\Desktop\NotDelete\GitHub\kaggleCompe_MAP-math\Vertex_AI_実験\インスタンスつくってみる\acemath-7b-qlora-improved"
    OUTPUT_PATH = r"C:\Users\mouse\Desktop\NotDelete\GitHub\kaggleCompe_MAP-math\Vertex_AI_実験\インスタンスつくってみる\kaggle-ready-acemath7b-qlora"

    # ラベルマッピング読み込み
    label_file = os.path.join(IMPROVED_MODEL_PATH, "label_mapping.json")
    if not os.path.exists(label_file):
        print(f"❌ ラベルマッピングが見つかりません: {label_file}")
        return None
    with open(label_file, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    print(f"✅ ラベル数: {len(label_mapping)}")

    # adapter_config読み込み
    adapter_config_path = os.path.join(IMPROVED_MODEL_PATH, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        print(f"❌ adapter_config.jsonが見つかりません: {adapter_config_path}")
        return None
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config.get(
        "base_model_name_or_path", "nvidia/AceMath-7B-Instruct"
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
    print("✅ ベースモデル読み込み完了")

    # QLoRAアダプター適用
    print("⚡ QLoRAアダプター読み込み中...")
    peft_model = PeftModel.from_pretrained(base_model, IMPROVED_MODEL_PATH)
    print("✅ QLoRAアダプター読み込み完了")

    # アダプターをベースモデルにマージ
    print("🔀 QLoRAアダプターをベースモデルにマージ中...")
    merged_model = peft_model.merge_and_unload()
    print("✅ マージ完了 - Kaggle用統合モデル作成成功")

    # トークナイザー読み込み
    print("📝 トークナイザー読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(IMPROVED_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ トークナイザー読み込み完了")

    # 出力ディレクトリ作成
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"📁 出力ディレクトリ作成: {OUTPUT_PATH}")

    # 統合モデル保存
    print(f"💾 統合モデル保存中...")
    merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print("✅ モデルとトークナイザー保存完了")

    # ラベルマッピングコピー
    import shutil

    shutil.copy2(label_file, os.path.join(OUTPUT_PATH, "label_mapping.json"))
    print("✅ ラベルマッピングコピー完了")

    print(f"\n🎯 Kaggleでのモデル使用方法:")
    print(f"1. {OUTPUT_PATH} フォルダをKaggleデータセットとしてアップロード")
    print(f"2. ノートブックでの読み込み例:")
    print(
        f"   from transformers import AutoModelForSequenceClassification, AutoTokenizer"
    )
    print(f"   import json")
    print(
        f"   model = AutoModelForSequenceClassification.from_pretrained('/kaggle/input/your-dataset-name')"
    )
    print(
        f"   tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/your-dataset-name')"
    )
    print(
        f"   with open('/kaggle/input/your-dataset-name/label_mapping.json', 'r') as f:"
    )
    print(f"       label_mapping = json.load(f)")
    print(f"✨ AceMath-7B QLoRAモデルのKaggle用統合が完了しました！")

    return OUTPUT_PATH


if __name__ == "__main__":
    result = merge_acemath7b_qlora_for_kaggle()
    if result:
        print(f"\n🚀 成功: {result}")
    else:
        print(f"\n💔 失敗: モデル統合に失敗しました")
