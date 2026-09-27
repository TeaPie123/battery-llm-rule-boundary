from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def iter_batches(path: Path, batch_size: int, limit: int | None):
    batch = []
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            batch.append(json.loads(line))
            seen += 1
            if len(batch) == batch_size:
                yield batch
                batch = []
            if limit is not None and seen >= limit:
                break
    if batch:
        yield batch


def candidate_log_probs(
    model,
    tokenizer,
    prompts: list[str],
    candidate: str,
    max_length: int,
) -> torch.Tensor:
    candidate_ids = tokenizer(
        candidate,
        add_special_tokens=False,
    )["input_ids"]
    if not candidate_ids:
        raise ValueError(f"Candidate has no tokens: {candidate!r}")

    sequences = []
    candidate_starts = []
    for prompt in prompts:
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - len(candidate_ids),
        )["input_ids"]
        candidate_starts.append(len(prompt_ids))
        sequences.append(prompt_ids + candidate_ids)

    max_sequence_length = max(len(sequence) for sequence in sequences)
    input_ids = []
    attention_mask = []
    for sequence in sequences:
        pad_length = max_sequence_length - len(sequence)
        input_ids.append(sequence + [tokenizer.pad_token_id] * pad_length)
        attention_mask.append([1] * len(sequence) + [0] * pad_length)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=model.device)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=model.device)

    with torch.inference_mode():
        logits = model(input_ids=input_tensor, attention_mask=mask_tensor).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)

    scores = []
    for row, start in enumerate(candidate_starts):
        score = torch.zeros((), device=model.device)
        for offset, token_id in enumerate(candidate_ids):
            token_position = start + offset
            if token_position == 0:
                raise ValueError("Candidate cannot start at position zero")
            score = score + log_probs[row, token_position - 1, token_id]
        scores.append(score / len(candidate_ids))
    return torch.stack(scores)


def one_token_class_log_probs(
    model,
    tokenizer,
    prompts: list[str],
    normal_token_id: int,
    anomaly_token_id: int,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [
        tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        for prompt in prompts
    ]
    max_sequence_length = max(len(sequence) for sequence in sequences)
    input_ids = []
    attention_mask = []
    last_positions = []
    for sequence in sequences:
        if not sequence:
            raise ValueError("Prompt has no tokens")
        pad_length = max_sequence_length - len(sequence)
        input_ids.append(sequence + [tokenizer.pad_token_id] * pad_length)
        attention_mask.append([1] * len(sequence) + [0] * pad_length)
        last_positions.append(len(sequence) - 1)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=model.device)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=model.device)
    row_indices = torch.arange(len(sequences), device=model.device)
    position_tensor = torch.tensor(
        last_positions, dtype=torch.long, device=model.device
    )
    with torch.inference_mode():
        logits = model(input_ids=input_tensor, attention_mask=mask_tensor).logits
        next_token_log_probs = torch.log_softmax(
            logits[row_indices, position_tensor].float(), dim=-1
        )
    return (
        next_token_log_probs[:, normal_token_id],
        next_token_log_probs[:, anomaly_token_id],
    )


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    adapter_path = args.adapter.resolve()
    test_path = args.test.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_model_path = project / "models/Qwen/Qwen2___5-0___5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path).cuda().eval()

    normal_tokens = tokenizer("正常", add_special_tokens=False)["input_ids"]
    anomaly_tokens = tokenizer("异常", add_special_tokens=False)["input_ids"]
    print(f"normal_tokens={normal_tokens}")
    print(f"anomaly_tokens={anomaly_tokens}")
    if len(normal_tokens) != 1 or len(anomaly_tokens) != 1:
        raise ValueError("Fast evaluator requires one token per class label")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    gold_values: list[int] = []
    score_values: list[float] = []
    rows = []

    for batch_number, batch in enumerate(
        iter_batches(test_path, args.batch_size, args.limit), 1
    ):
        prompts = [
            tokenizer.apply_chat_template(
                item["messages"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in batch
        ]
        normal_log_prob, anomaly_log_prob = one_token_class_log_probs(
            model,
            tokenizer,
            prompts,
            normal_tokens[0],
            anomaly_tokens[0],
            args.max_length,
        )
        anomaly_scores = torch.sigmoid(anomaly_log_prob - normal_log_prob)
        for item, score in zip(batch, anomaly_scores.detach().cpu().tolist()):
            if "is_anomaly" in item:
                gold_value = item["is_anomaly"]
            elif "gold" in item:
                gold_value = item["gold"]
            else:
                raise KeyError("Sample has neither 'is_anomaly' nor 'gold'")
            gold = int(bool(gold_value))
            prediction = int(score >= 0.5)
            gold_values.append(gold)
            score_values.append(float(score))
            rows.append(
                {
                    "record_id": item.get("record_id"),
                    "cycle_no": item.get("cycle_no"),
                    "case_type": item.get("case_type"),
                    "gold": bool(gold),
                    "pred": bool(prediction),
                    "anomaly_score": float(score),
                }
            )
        if batch_number % 100 == 0:
            print(f"processed={len(gold_values)}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    gold_array = np.asarray(gold_values, dtype=np.int64)
    score_array = np.asarray(score_values, dtype=np.float64)
    pred_array = (score_array >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        gold_array, pred_array, labels=[0, 1]
    ).ravel()
    metrics = {
        "adapter": str(adapter_path),
        "test": str(test_path),
        "threshold": 0.5,
        "samples": int(len(gold_array)),
        "prevalence": float(gold_array.mean()),
        "accuracy": float(accuracy_score(gold_array, pred_array)),
        "precision": float(precision_score(gold_array, pred_array, zero_division=0)),
        "recall": float(recall_score(gold_array, pred_array, zero_division=0)),
        "f1": float(f1_score(gold_array, pred_array, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "auroc": float(roc_auc_score(gold_array, score_array)),
        "auprc": float(average_precision_score(gold_array, score_array)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "elapsed_seconds": elapsed,
        "samples_per_second": len(gold_array) / elapsed if elapsed else math.inf,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with output_path.with_suffix(".metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
