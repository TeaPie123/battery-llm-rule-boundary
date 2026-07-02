import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

MODEL_DIR = "models/Qwen/Qwen2___5-0___5B-Instruct"
TRAIN_PATH = "data/processed/lora_full_balanced_v3/train.jsonl"
OUT_DIR = "outputs/qwen_lora_v3_full_balanced"
MAX_LENGTH = 1024

os.makedirs(OUT_DIR, exist_ok=True)

print("模型路径:", MODEL_DIR)
print("训练集:", TRAIN_PATH)
print("输出目录:", OUT_DIR)
print("cuda available:", torch.cuda.is_available())

bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if bf16_ok else torch.float16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=dtype,
    trust_remote_code=True
)

if torch.cuda.is_available():
    model = model.cuda()

model.config.use_cache = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

class BatteryDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print("读取训练样本数:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]

        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        )

        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH
        )["input_ids"]

        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for x in batch:
        pad_len = max_len - len(x["input_ids"])

        input_ids.append(x["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


train_dataset = BatteryDataset(TRAIN_PATH)

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    max_steps=1000,
    logging_steps=20,
    save_steps=200,
    save_total_limit=2,
    bf16=bf16_ok,
    fp16=(torch.cuda.is_available() and not bf16_ok),
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

print("开始 LoRA 小样本训练...")
trainer.train()

print("保存 LoRA adapter:", OUT_DIR)
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("训练完成")
