import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType


BASE_MODEL = "models/Qwen/Qwen2___5-0___5B-Instruct"
TRAIN_PATH = "data/processed/lora_boundary_v4/train_clean.jsonl"
VAL_PATH = "data/processed/lora_boundary_v4/val_clean.jsonl"
OUT_DIR = "outputs/qwen_lora_v4_boundary"

MAX_LENGTH = 1024


def format_example(example, tokenizer):
    messages = example["messages"]

    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )

    answer = messages[-1]["content"]
    full_text = prompt + answer + tokenizer.eos_token

    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    tokenized_prompt = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]

    prompt_len = len(tokenized_prompt["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids = []
        attention_mask = []
        labels = []

        pad_id = self.tokenizer.pad_token_id

        for x in features:
            length = len(x["input_ids"])
            pad_len = max_len - length

            input_ids.append(x["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    print("BASE_MODEL:", BASE_MODEL)
    print("TRAIN_PATH:", TRAIN_PATH)
    print("VAL_PATH:", VAL_PATH)
    print("OUT_DIR:", OUT_DIR)
    print("cuda available:", torch.cuda.is_available())

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_PATH,
            "validation": VAL_PATH,
        },
    )

    tokenized = dataset.map(
        lambda x: format_example(x, tokenizer),
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_steps=1200,
        logging_steps=20,
        save_steps=300,
        eval_steps=300,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForCausalLM(tokenizer),
    )

    trainer.train()

    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print("v4 training finished.")
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
