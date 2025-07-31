import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "nvidia/AceMath-7B-Instruct"
NUM_LABELS = 65

# QLoRA量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# トークナイザー読み込み
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✅ Tokenizer loaded")

# モデル設定
config = AutoConfig.from_pretrained(MODEL_NAME)
config.num_labels = NUM_LABELS
config.problem_type = "single_label_classification"

# モデル読み込み
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
print("✅ Model loaded")

# QLoRA適用
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
print("✅ QLoRA applied")

print("All steps completed successfully.")