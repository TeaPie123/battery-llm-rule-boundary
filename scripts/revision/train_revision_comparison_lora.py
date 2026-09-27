from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


class BatteryDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int, limit: int | None = None):
        self.items = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.items.append(json.loads(line))
                    if limit is not None and len(self.items) >= limit:
                        break
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        messages = item["messages"]
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False
        )["input_ids"]
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        labels = full_ids.copy()
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(
            len(prompt_ids), len(labels)
        )
        if not any(label != -100 for label in labels):
            raise ValueError(f"Sample {index} has no supervised response tokens")
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "sample_weight": float(item.get("sample_weight", 1.0)),
        }


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in batch)
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            pad_length = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_length)
            attention_mask.append(item["attention_mask"] + [0] * pad_length)
            labels.append(item["labels"] + [-100] * pad_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_weight": torch.tensor(
                [item["sample_weight"] for item in batch], dtype=torch.float32
            ),
        }


class SampleWeightedTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        sample_weight = inputs.pop("sample_weight")
        labels = inputs["labels"]
        outputs = model(**inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)
        supervised = shift_labels.ne(-100)
        per_sample = (token_loss * supervised).sum(dim=1) / supervised.sum(
            dim=1
        ).clamp_min(1)
        loss = (per_sample * sample_weight.to(per_sample.device)).mean()
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    train_path = args.train.resolve()
    val_path = args.val.resolve()
    output = args.output.resolve()
    base_model_path = project / "models/Qwen/Qwen2___5-0___5B-Instruct"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, dtype=dtype, trust_remote_code=True
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = BatteryDataset(train_path, tokenizer, args.max_length)
    val_dataset = BatteryDataset(
        val_path, tokenizer, args.max_length, 4 if args.dry_run else None
    )
    if args.dry_run:
        highest_weight_item = max(
            train_dataset.items,
            key=lambda item: float(item.get("sample_weight", 1.0)),
        )
        unit_weight_items = [
            item
            for item in train_dataset.items
            if float(item.get("sample_weight", 1.0)) == 1.0
        ][:3]
        train_dataset.items = [highest_weight_item] + unit_weight_items
    weights = [
        float(item.get("sample_weight", 1.0)) for item in train_dataset.items
    ]
    resolved_config = {
        "experiment": args.experiment,
        "base_model": str(base_model_path),
        "train": str(train_path),
        "val": str(val_path),
        "output": str(output),
        "seed": args.seed,
        "max_steps": 1 if args.dry_run else args.max_steps,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "sample_weight": {
            "min": min(weights),
            "max": max(weights),
            "sum": sum(weights),
            "weighted_samples": sum(weight != 1.0 for weight in weights),
            "loss_reduction": "mean(per_sample_token_loss * sample_weight)",
        },
        "lora": {
            "r": 8,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": sorted(lora_config.target_modules),
        },
        "batch": {
            "per_device": 1,
            "gradient_accumulation": 1 if args.dry_run else 8,
            "effective": 1 if args.dry_run else 8,
        },
        "precision": str(dtype),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with (output / "resolved_training_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(resolved_config, handle, ensure_ascii=False, indent=2)

    training_args = TrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if args.dry_run else 8,
        learning_rate=args.learning_rate,
        max_steps=1 if args.dry_run else args.max_steps,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        logging_steps=1 if args.dry_run else 20,
        eval_strategy="no" if args.dry_run else "steps",
        eval_steps=250,
        save_strategy="no" if args.dry_run else "steps",
        save_steps=250,
        save_total_limit=2,
        load_best_model_at_end=False,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
    )
    trainer = SampleWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None if args.dry_run else val_dataset,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    trainer.train()
    if args.dry_run:
        print("DRY_RUN_OK")
        return
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"TRAINING_COMPLETE={output}")


if __name__ == "__main__":
    main()
