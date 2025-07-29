import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# GPUが利用可能か確認
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# モデルとトークナイザーのパス
model_id = "nvidia/AceMath-7B-Instruct"

print(f"Loading model: {model_id}...")
# モデルとトークナイザーをロード
# L4 GPUのメモリに収まるか確認
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("Model loaded.")

# 推論プロンプト
prompt = "What is pi?"

# プロンプトをトークナイズ
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("Generating response...")
# テキスト生成
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50, # 生成する最大の新しいトークン数
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

# 生成されたテキストをデコード
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- Response ---")
print(response)
print("----------------")
