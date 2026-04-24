from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

LOCAL_BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TARGET_MODULES = ["q_proj", "v_proj"]

def build_lora_model():
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_BASE_MODEL_ID,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, config)
    return model

def save_adapter(path: str):
    model = build_lora_model()
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)

save_adapter("adapters/analytic")
save_adapter("adapters/engagement")
print("Saved starter LoRAs.")