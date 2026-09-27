from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
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


SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def iter_batches(rows: list[dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def stable_example_key(record_id: int) -> str:
    return hashlib.sha256(f"{SEED}:{record_id}".encode()).hexdigest()


def compact_demo_user(item: dict) -> str:
    prompt = item["messages"][1]["content"]
    prefixes = [
        "状态编码：",
        "当前总电压：",
        "本循环上一行总电压：",
        "总电压相邻跳变：",
        "当前单体电压：",
        "本循环上一行单体电压：",
        "单体电压相邻跳变：",
        "当前电流：",
        "本循环上一行电流：",
        "电流相邻跳变：",
        "当前温度：",
        "本循环上一行温度：",
        "温度相邻跳变：",
    ]
    extracted = [
        line.strip()
        for line in prompt.splitlines()
        if any(line.strip().startswith(prefix) for prefix in prefixes)
    ]
    if len(extracted) != len(prefixes):
        raise ValueError(
            f"Could not extract compact demonstration fields: {item['record_id']}"
        )
    return (
        "按照系统给出的五条电池异常规则判断以下训练示例，只回答正常或异常。\n"
        + "\n".join(extracted)
    )


def select_demonstrations(train_path: Path) -> list[dict]:
    rows = read_jsonl(train_path)
    by_label = {False: [], True: []}
    for item in rows:
        by_label[bool(item["is_anomaly"])].append(item)
    selected = []
    for label in [False, True]:
        candidates = sorted(
            by_label[label],
            key=lambda item: stable_example_key(int(item["record_id"])),
        )
        selected.extend(candidates[:2])
    random.Random(SEED).shuffle(selected)
    result = []
    for item in selected:
        label = bool(item["is_anomaly"])
        result.append(
            {
                "record_id": int(item["record_id"]),
                "cycle_no": str(item["cycle_no"]),
                "label": label,
                "user": compact_demo_user(item),
                "assistant": "异常" if label else "正常",
                "selection_key": stable_example_key(int(item["record_id"])),
            }
        )
    return result


def build_prompt_messages(
    item: dict, mode: str, demonstrations: list[dict]
) -> list[dict]:
    target_messages = item["messages"][:-1]
    if len(target_messages) != 2:
        raise ValueError("Expected exactly system + user before target answer")
    if mode == "zero_shot":
        return target_messages
    messages = [target_messages[0]]
    for demo in demonstrations:
        messages.append({"role": "user", "content": demo["user"]})
        messages.append({"role": "assistant", "content": demo["assistant"]})
    messages.append(target_messages[1])
    return messages


def class_log_probs_last_position(
    model,
    tokenizer,
    prompts: list[str],
    normal_token_id: int,
    anomaly_token_id: int,
    max_length: int,
    logits_argument: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    sequences = [
        tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for prompt in prompts
    ]
    original_lengths = [len(sequence) for sequence in sequences]
    if max(original_lengths) > max_length:
        raise ValueError(
            f"Prompt exceeds max_length without an explicit truncation protocol: "
            f"max={max(original_lengths)} limit={max_length}"
        )
    max_sequence_length = max(original_lengths)
    input_ids = []
    attention_mask = []
    for sequence in sequences:
        pad_length = max_sequence_length - len(sequence)
        input_ids.append([tokenizer.pad_token_id] * pad_length + sequence)
        attention_mask.append([0] * pad_length + [1] * len(sequence))
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=model.device)
    mask_tensor = torch.tensor(
        attention_mask, dtype=torch.long, device=model.device
    )
    model_kwargs = {
        "input_ids": input_tensor,
        "attention_mask": mask_tensor,
    }
    model_kwargs[logits_argument] = 1
    with torch.inference_mode():
        logits = model(**model_kwargs).logits
        if logits.shape[1] != 1:
            raise ValueError(
                f"Last-logit optimization did not take effect: {logits.shape}"
            )
        next_token_log_probs = torch.log_softmax(
            logits[:, -1, :].float(), dim=-1
        )
    return (
        next_token_log_probs[:, normal_token_id],
        next_token_log_probs[:, anomaly_token_id],
        original_lengths,
    )


def evaluate(
    model,
    tokenizer,
    rows: list[dict],
    mode: str,
    demonstrations: list[dict],
    batch_size: int,
    max_length: int,
    normal_token_id: int,
    anomaly_token_id: int,
    logits_argument: str,
) -> tuple[list[dict], dict, dict]:
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output_rows = []
    gold_values = []
    score_values = []
    prompt_lengths = []

    for batch_number, batch in enumerate(iter_batches(rows, batch_size), 1):
        prompts = [
            tokenizer.apply_chat_template(
                build_prompt_messages(item, mode, demonstrations),
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in batch
        ]
        normal_log_prob, anomaly_log_prob, lengths = (
            class_log_probs_last_position(
                model,
                tokenizer,
                prompts,
                normal_token_id,
                anomaly_token_id,
                max_length,
                logits_argument,
            )
        )
        anomaly_scores = torch.sigmoid(anomaly_log_prob - normal_log_prob)
        prompt_lengths.extend(lengths)
        for item, score in zip(batch, anomaly_scores.cpu().tolist()):
            if "is_anomaly" in item:
                gold = bool(item["is_anomaly"])
            elif "gold" in item:
                gold = bool(item["gold"])
            else:
                raise KeyError("Sample lacks gold label")
            score = float(score)
            gold_values.append(int(gold))
            score_values.append(score)
            output_rows.append(
                {
                    "record_id": item.get("record_id"),
                    "cycle_no": item.get("cycle_no"),
                    "case_type": item.get("case_type"),
                    "gold": gold,
                    "pred": bool(score >= 0.5),
                    "anomaly_score": score,
                }
            )
        if batch_number % 100 == 0:
            print(
                f"mode={mode} processed={len(output_rows)} "
                f"max_prompt_tokens={max(prompt_lengths)}",
                flush=True,
            )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    gold_array = np.asarray(gold_values, dtype=np.int64)
    score_array = np.asarray(score_values, dtype=np.float64)
    pred_array = (score_array >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        gold_array, pred_array, labels=[0, 1]
    ).ravel()
    metrics = {
        "threshold": 0.5,
        "samples": int(gold_array.size),
        "prevalence": float(gold_array.mean()),
        "accuracy": float(accuracy_score(gold_array, pred_array)),
        "precision": float(
            precision_score(gold_array, pred_array, zero_division=0)
        ),
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
        "samples_per_second": gold_array.size / elapsed,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    length_summary = {
        "min": int(min(prompt_lengths)),
        "max": int(max(prompt_lengths)),
        "mean": float(np.mean(prompt_lengths)),
        "p95": float(np.quantile(prompt_lengths, 0.95)),
        "p99": float(np.quantile(prompt_lengths, 0.99)),
        "truncated": 0,
    }
    return output_rows, metrics, length_summary


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    final_dir = project / "outputs" / args.output_name
    output_dir = project / "outputs" / f"{args.output_name}.tmp"
    if final_dir.exists() or output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {final_dir} or {output_dir}")
    output_dir.mkdir(parents=True)

    model_path = project / "models/Qwen/Qwen2___5-0___5B-Instruct"
    grouped_dir = project / "data/processed/revision_grouped_v1"
    boundary_path = project / "data/processed/boundary_test/boundary_test.jsonl"
    datasets = {
        "balanced": grouped_dir / "test_balanced.jsonl",
        "natural": grouped_dir / "test_natural.jsonl",
        "boundary": boundary_path,
    }
    demonstrations = select_demonstrations(grouped_dir / "train_balanced.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).cuda().eval()
    signature = inspect.signature(model.forward)
    if "logits_to_keep" in signature.parameters:
        logits_argument = "logits_to_keep"
    elif "num_logits_to_keep" in signature.parameters:
        logits_argument = "num_logits_to_keep"
    else:
        raise RuntimeError(
            "Model does not support last-position-only logits; refusing slow "
            "full-sequence evaluation."
        )

    normal_tokens = tokenizer("正常", add_special_tokens=False)["input_ids"]
    anomaly_tokens = tokenizer("异常", add_special_tokens=False)["input_ids"]
    if len(normal_tokens) != 1 or len(anomaly_tokens) != 1:
        raise ValueError("Class labels are not single tokens")

    manifest = {
        "version": args.output_name,
        "seed": SEED,
        "model": str(model_path),
        "adapter": None,
        "modes": {
            "zero_shot": "Target system and user messages only.",
            "few_shot": (
                "Four deterministic compact demonstrations selected from the "
                "balanced training split: two normal and two anomaly."
            ),
        },
        "demonstrations": demonstrations,
        "selection_policy": (
            "Within each label, select the two smallest SHA-256 keys of "
            "'seed:record_id', then shuffle the four with seed 42."
        ),
        "classification": {
            "normal_token_id": normal_tokens[0],
            "anomaly_token_id": anomaly_tokens[0],
            "score": "sigmoid(logP(anomaly)-logP(normal))",
            "threshold": 0.5,
            "logits_optimization": logits_argument,
        },
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "limit": args.limit,
        "datasets": {},
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }

    comparison_rows = []
    for dataset_name, dataset_path in datasets.items():
        source_rows = read_jsonl(dataset_path, args.limit)
        manifest["datasets"][dataset_name] = {
            "path": str(dataset_path),
            "evaluated_rows": len(source_rows),
        }
        for mode in ["zero_shot", "few_shot"]:
            predictions, metrics, prompt_lengths = evaluate(
                model,
                tokenizer,
                source_rows,
                mode,
                demonstrations,
                args.batch_size,
                args.max_length,
                normal_tokens[0],
                anomaly_tokens[0],
                logits_argument,
            )
            prefix = f"{mode}_{dataset_name}"
            with (output_dir / f"{prefix}.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                for row in predictions:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics.update(
                {
                    "mode": mode,
                    "test": str(dataset_path),
                    "model": str(model_path),
                    "adapter": None,
                    "prompt_lengths": prompt_lengths,
                }
            )
            (output_dir / f"{prefix}.metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifest.setdefault("results", {}).setdefault(mode, {})[
                dataset_name
            ] = metrics
            comparison_rows.append(
                {
                    "mode": mode,
                    "dataset": dataset_name,
                    **{
                        key: metrics[key]
                        for key in [
                            "samples",
                            "prevalence",
                            "accuracy",
                            "precision",
                            "recall",
                            "f1",
                            "fpr",
                            "fnr",
                            "specificity",
                            "auroc",
                            "auprc",
                            "tp",
                            "fp",
                            "tn",
                            "fn",
                            "elapsed_seconds",
                            "samples_per_second",
                            "peak_memory_mib",
                        ]
                    },
                }
            )

    with (output_dir / "comparison_table.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    provenance = {"version": args.output_name, "files": {}}
    for path in sorted(output_dir.iterdir()):
        if path.name == "provenance_manifest.json" or not path.is_file():
            continue
        provenance["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    (output_dir / "provenance_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_dir.rename(final_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"OUTPUT_DIR={final_dir}")


if __name__ == "__main__":
    main()
