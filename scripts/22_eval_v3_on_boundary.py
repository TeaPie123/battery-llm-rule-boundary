import json
import re
import torch
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "models/Qwen/Qwen2___5-0___5B-Instruct"
LORA_DIR = "outputs/qwen_lora_v3_full_balanced"
TEST_PATH = Path("data/processed/boundary_test/boundary_test.jsonl")

OUT_DIR = Path("outputs/boundary_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PRED = OUT_DIR / "v3_boundary_predictions.jsonl"
OUT_CSV = OUT_DIR / "v3_boundary_cases.csv"
OUT_WRONG = OUT_DIR / "v3_boundary_wrong_cases.csv"
OUT_METRICS = OUT_DIR / "v3_boundary_metrics.txt"

BATCH_SIZE = 8
MAX_NEW_TOKENS = 100


def parse_pred(text):
    if text is None:
        return None
    s = text.strip()
    if s.startswith("异常"):
        return True
    if s.startswith("正常"):
        return False
    first = s[:30]
    if "异常" in first and "正常" not in first:
        return True
    if "正常" in first and "异常" not in first:
        return False
    return None


def first_float(patterns, text):
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def first_int(patterns, text):
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def rule_verify(user_text):
    state = first_int([r"状态编码[:：]\s*(-?\d+)"], user_text)

    total_voltage = first_float([r"当前总电压[:：]\s*([-+]?\d+(?:\.\d+)?)"], user_text)
    total_diff = first_float([r"总电压相邻跳变[:：]\s*([-+]?\d+(?:\.\d+)?)"], user_text)
    cell_diff = first_float([r"单体电压相邻跳变[:：]\s*([-+]?\d+(?:\.\d+)?)"], user_text)
    current_diff = first_float([r"电流相邻跳变[:：]\s*([-+]?\d+(?:\.\d+)?)"], user_text)
    temp_diff = first_float([r"温度相邻跳变[:：]\s*([-+]?\d+(?:\.\d+)?)"], user_text)

    reasons = []

    if state == 110:
        if total_diff is not None and total_diff > 3.0:
            reasons.append(f"充电总电压跳变异常：{total_diff:.4f}V > 3.0V")
        if current_diff is not None and current_diff > 0.5:
            reasons.append(f"充电电流跳变异常：{current_diff:.4f}A > 0.5A")

    # 与当前真实标签逻辑一致：单体电压和温度跳变是通用规则
    if cell_diff is not None and cell_diff > 0.05:
        reasons.append(f"单体电压跳变异常：{cell_diff:.4f}V > 0.05V")
    if temp_diff is not None and temp_diff > 3.0:
        reasons.append(f"温度跳变异常：{temp_diff:.4f}℃ > 3.0℃")

    if state == 30:
        if total_voltage is not None and total_voltage > 378.2:
            reasons.append(f"放电总电压超限：{total_voltage:.4f}V > 378.2V")

    return len(reasons) > 0, "；".join(reasons) if reasons else "无异常"


def calc_metrics(rows, key):
    valid = [r for r in rows if r[key] is not None]
    total = len(valid)
    correct = sum(r[key] == r["gold"] for r in valid)

    tp = sum(r["gold"] is True and r[key] is True for r in valid)
    fp = sum(r["gold"] is False and r[key] is True for r in valid)
    tn = sum(r["gold"] is False and r[key] is False for r in valid)
    fn = sum(r["gold"] is True and r[key] is False for r in valid)

    acc = correct / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def build_prompt(tokenizer, item):
    return tokenizer.apply_chat_template(
        item["messages"][:-1],
        tokenize=False,
        add_generation_prompt=True,
    )


print("基础模型:", BASE_MODEL, flush=True)
print("LoRA:", LORA_DIR, flush=True)
print("BoundarySet:", TEST_PATH, flush=True)
print("cuda available:", torch.cuda.is_available(), flush=True)

records = []
with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print("BoundarySet samples:", len(records), flush=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, LORA_DIR)

if torch.cuda.is_available():
    model = model.cuda()

model.eval()

rows = []

with open(OUT_PRED, "w", encoding="utf-8") as fout:
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        prompts = [build_prompt(tokenizer, item) for item in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]

        for item, output_ids in zip(batch, outputs):
            gen_ids = output_ids[input_len:]
            pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            llm_pred = parse_pred(pred_text)
            gold = bool(item["gold"])
            user_text = item["messages"][1]["content"]

            rule_pred, rule_reason = rule_verify(user_text)

            if llm_pred is None:
                final_pred = rule_pred
                correction_type = "llm_unparsed_use_rule"
            elif llm_pred == rule_pred:
                final_pred = llm_pred
                correction_type = "consistent"
            else:
                final_pred = rule_pred
                correction_type = "corrected_by_rule"

            row = {
                "record_id": item["record_id"],
                "case_type": item["case_type"],
                "gold": gold,
                "gold_text": item["messages"][-1]["content"],
                "pred_text": pred_text,
                "llm_pred": llm_pred,
                "rule_pred": rule_pred,
                "final_pred": final_pred,
                "llm_correct": llm_pred == gold if llm_pred is not None else False,
                "rule_correct": rule_pred == gold,
                "final_correct": final_pred == gold,
                "llm_rule_consistent": llm_pred == rule_pred if llm_pred is not None else False,
                "correction_type": correction_type,
                "rule_reason": rule_reason,
                "state": item["state"],
                "total_voltage_diff": item["total_voltage_diff"],
                "cell_voltage_diff": item["cell_voltage_diff"],
                "current_diff": item["current_diff"],
                "temperature_diff": item["temperature_diff"],
            }

            rows.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"进度: {min(start + BATCH_SIZE, len(records))}/{len(records)}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

df_wrong = df[df["llm_correct"] == False].copy()
df_wrong.to_csv(OUT_WRONG, index=False, encoding="utf-8-sig")

llm_metrics = calc_metrics(rows, "llm_pred")
rule_metrics = calc_metrics(rows, "rule_pred")
final_metrics = calc_metrics(rows, "final_pred")

case_lines = []
case_lines.append("BoundarySet case_type accuracy:")
for case_type, g in df.groupby("case_type"):
    acc = g["llm_correct"].mean()
    total = len(g)
    wrong = int((g["llm_correct"] == False).sum())
    case_lines.append(f"{case_type}: total={total}, wrong={wrong}, acc={acc}")

summary = f"""v3 on BoundarySet evaluation

BoundarySet: {TEST_PATH}
Total samples: {len(rows)}

LLM-only:
Accuracy: {llm_metrics['accuracy']}
Precision: {llm_metrics['precision']}
Recall: {llm_metrics['recall']}
F1: {llm_metrics['f1']}
TP: {llm_metrics['tp']}
FP: {llm_metrics['fp']}
TN: {llm_metrics['tn']}
FN: {llm_metrics['fn']}

RuleVerifier-only:
Accuracy: {rule_metrics['accuracy']}
Precision: {rule_metrics['precision']}
Recall: {rule_metrics['recall']}
F1: {rule_metrics['f1']}
TP: {rule_metrics['tp']}
FP: {rule_metrics['fp']}
TN: {rule_metrics['tn']}
FN: {rule_metrics['fn']}

LLM + RuleVerifier:
Accuracy: {final_metrics['accuracy']}
Precision: {final_metrics['precision']}
Recall: {final_metrics['recall']}
F1: {final_metrics['f1']}
TP: {final_metrics['tp']}
FP: {final_metrics['fp']}
TN: {final_metrics['tn']}
FN: {final_metrics['fn']}

LLM wrong cases: {len(df_wrong)}
LLM-rule inconsistent cases: {(df['llm_rule_consistent'] == False).sum()}
Corrected by rule cases: {(df['correction_type'] == 'corrected_by_rule').sum()}

{chr(10).join(case_lines)}

Output files:
{OUT_PRED}
{OUT_CSV}
{OUT_WRONG}
{OUT_METRICS}
"""

OUT_METRICS.write_text(summary, encoding="utf-8")
print(summary)

if len(df_wrong) > 0:
    print("LLM 错误样本预览:")
    cols = [
        "record_id",
        "case_type",
        "gold",
        "llm_pred",
        "rule_pred",
        "final_pred",
        "current_diff",
        "cell_voltage_diff",
        "temperature_diff",
        "total_voltage_diff",
        "pred_text",
        "gold_text",
    ]
    print(df_wrong[cols].to_string(index=False))
