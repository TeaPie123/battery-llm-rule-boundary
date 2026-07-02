import json
import re
from pathlib import Path
import pandas as pd

TEST_PATH = Path("data/processed/lora_full_balanced_v3/test.jsonl")
PRED_PATH = Path("outputs/qwen_lora_v3_full_balanced/eval_full_batched_predictions.jsonl")

OUT_DIR = Path("outputs/rule_verifier")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CASES = OUT_DIR / "rule_verifier_cases.csv"
OUT_INCONSISTENT = OUT_DIR / "llm_rule_inconsistent_cases.csv"
OUT_METRICS = OUT_DIR / "rule_verifier_metrics.txt"


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


def parse_record_id(text):
    m = re.search(r"记录号：(\d+)", text)
    return m.group(1) if m else None


def abs_diff(a, b):
    if a is None or b is None:
        return None
    return abs(a - b)


def rule_verify(user_text):
    """
    严格版 RuleVerifier：
    1. 优先解析 prompt 中的显式跳变值；
    2. 如果显式跳变值没有解析到，则从上一行值和当前值重新计算；
    3. 使用师兄 new_error.py 当前复现实验中的阈值。
    """

    state = first_int([
        r"状态编码[:：]\s*(-?\d+)",
        r"整车State状态（状态机编码）[:：]\s*(-?\d+)",
        r"State状态[:：]\s*(-?\d+)",
    ], user_text)

    # 总电压
    prev_total_voltage = first_float([
        r"上一行总电压[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"上一行动力电池内部总电压V1[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"上一时刻总电压[:：]\s*([-+]?\d+(?:\.\d+)?)",
    ], user_text)

    current_total_voltage = first_float([
        r"当前总电压[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"本行总电压[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"当前动力电池内部总电压V1[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"本行动力电池内部总电压V1[:：]\s*([-+]?\d+(?:\.\d+)?)",
    ], user_text)

    total_voltage_diff = first_float([
        r"总电压相邻跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"总电压跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"动力电池内部总电压V1相邻跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"动力电池内部总电压V1跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
    ], user_text)
    if total_voltage_diff is None:
        total_voltage_diff = abs_diff(current_total_voltage, prev_total_voltage)

    # 单体电压
    prev_cell_voltage = first_float([
        r"上一行1号电池单体电压[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"上一行单体电压[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"上一时刻1号电池单体电压[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"上一时刻单体电压[:：]\s*([-+]?\d+(?:\.\d+)?)",
    ], user_text)

    current_cell_voltage = first_float([
        r"当前1号电池单体电压[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"本行1号电池单体电压[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"当前单体电压[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"本行单体电压[:：]\s*([-+]?\d+(?:\.\d+)?)",
    ], user_text)

    cell_voltage_diff = first_float([
        r"1号电池单体电压相邻跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"1号电池单体电压跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"单体电压相邻跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"单体电压跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
    ], user_text)
    if cell_voltage_diff is None:
        cell_voltage_diff = abs_diff(current_cell_voltage, prev_cell_voltage)

    # 电流
    prev_current = first_float([
        r"上一行电流[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"上一行动力电池充/放电电流[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"上一时刻电流[:：]\s*([-+]?\d+(?:\.\d+)?)",
    ], user_text)

    current_current = first_float([
        r"当前电流[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"本行电流[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"当前动力电池充/放电电流[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"本行动力电池充/放电电流[:：]\s*([-+]?\d+(?:\.\d+)?)",
    ], user_text)

    current_diff = first_float([
        r"电流相邻跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"电流跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"动力电池充/放电电流相邻跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"动力电池充/放电电流跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
    ], user_text)
    if current_diff is None:
        current_diff = abs_diff(current_current, prev_current)

    # 温度
    prev_temp = first_float([
        r"上一行温度[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"上一行1号温度检测点温度[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"上一时刻温度[:：]\s*([-+]?\d+(?:\.\d+)?)",
    ], user_text)

    current_temp = first_float([
        r"当前温度[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"本行温度[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"当前1号温度检测点温度[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"本行1号温度检测点温度[:：]\s*([-+]?\d+(?:\.\d+)?)",
    ], user_text)

    temperature_diff = first_float([
        r"温度相邻跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"温度跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"1号温度检测点温度相邻跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"1号温度检测点温度跳变[:：]\s*([-+]?\d+(?:\.\d+)?)",
    ], user_text)
    if temperature_diff is None:
        temperature_diff = abs_diff(current_temp, prev_temp)

    reasons = []

    if state == 110:
        if total_voltage_diff is not None and total_voltage_diff > 3.0:
            reasons.append(f"充电总电压跳变异常：{total_voltage_diff:.6f}V > 3.0V")
        if current_diff is not None and current_diff > 0.5:
            reasons.append(f"充电电流跳变异常：{current_diff:.6f}A > 0.5A")

    # 与师兄 new_error.py 的实际标签逻辑对齐：
    # 单体电压跳变和温度跳变作为通用规则，不限制状态编码。
    if cell_voltage_diff is not None and cell_voltage_diff > 0.05:
        reasons.append(f"单体电压跳变异常：{cell_voltage_diff:.6f}V > 0.05V")
    if temperature_diff is not None and temperature_diff > 3.0:
        reasons.append(f"温度跳变异常：{temperature_diff:.6f}℃ > 3.0℃")

    if state == 30:
        if current_total_voltage is not None and current_total_voltage > 378.2:
            reasons.append(f"放电总电压超限：{current_total_voltage:.6f}V > 378.2V")

    parsed = {
        "state": state,
        "current_total_voltage": current_total_voltage,
        "total_voltage_diff": total_voltage_diff,
        "cell_voltage_diff": cell_voltage_diff,
        "current_diff": current_diff,
        "temperature_diff": temperature_diff,
    }

    return len(reasons) > 0, "；".join(reasons) if reasons else "无异常", parsed


def load_test_items():
    items = {}
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            user_text = item["messages"][1]["content"]
            rid = parse_record_id(user_text)
            if rid is not None:
                items[rid] = item
    return items


def calc_metrics(rows, key):
    valid_rows = [r for r in rows if r[key] is not None]
    total = len(valid_rows)
    correct = sum(r[key] == r["gold"] for r in valid_rows)

    tp = sum(r["gold"] is True and r[key] is True for r in valid_rows)
    fp = sum(r["gold"] is False and r[key] is True for r in valid_rows)
    tn = sum(r["gold"] is False and r[key] is False for r in valid_rows)
    fn = sum(r["gold"] is True and r[key] is False for r in valid_rows)

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


test_items = load_test_items()
rows = []

with open(PRED_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        pred_item = json.loads(line)
        rid = str(pred_item.get("record_id"))

        test_item = test_items.get(rid)
        if test_item is None:
            continue

        user_text = test_item["messages"][1]["content"]
        rule_pred, rule_reason, parsed = rule_verify(user_text)

        llm_pred = pred_item.get("pred")
        gold = bool(pred_item.get("gold"))

        if llm_pred is None:
            final_pred = rule_pred
            correction_type = "llm_unparsed_use_rule"
        elif llm_pred == rule_pred:
            final_pred = llm_pred
            correction_type = "consistent"
        else:
            final_pred = rule_pred
            correction_type = "corrected_by_rule"

        rows.append({
            "record_id": rid,
            "gold": gold,
            "llm_pred": llm_pred,
            "rule_pred": rule_pred,
            "final_pred": final_pred,
            "llm_correct": llm_pred == gold if llm_pred is not None else False,
            "rule_correct": rule_pred == gold,
            "final_correct": final_pred == gold,
            "llm_rule_consistent": llm_pred == rule_pred if llm_pred is not None else False,
            "correction_type": correction_type,
            "rule_reason": rule_reason,
            "state": parsed["state"],
            "current_total_voltage": parsed["current_total_voltage"],
            "total_voltage_diff": parsed["total_voltage_diff"],
            "cell_voltage_diff": parsed["cell_voltage_diff"],
            "current_diff": parsed["current_diff"],
            "temperature_diff": parsed["temperature_diff"],
            "gold_text": pred_item.get("gold_text"),
            "pred_text": pred_item.get("pred_text"),
            "user_text": user_text,
        })


df = pd.DataFrame(rows)
df.to_csv(OUT_CASES, index=False, encoding="utf-8-sig")

df_inconsistent = df[df["llm_rule_consistent"] == False].copy()
df_inconsistent.to_csv(OUT_INCONSISTENT, index=False, encoding="utf-8-sig")

llm_metrics = calc_metrics(rows, "llm_pred")
rule_metrics = calc_metrics(rows, "rule_pred")
final_metrics = calc_metrics(rows, "final_pred")

summary = f"""RuleVerifier v3 evaluation

Input test file:
{TEST_PATH}

Input prediction file:
{PRED_PATH}

Total matched samples: {len(rows)}

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

LLM-rule inconsistent cases: {len(df_inconsistent)}
Corrected by rule cases: {(df['correction_type'] == 'corrected_by_rule').sum()}

Output files:
{OUT_CASES}
{OUT_INCONSISTENT}
{OUT_METRICS}
"""

OUT_METRICS.write_text(summary, encoding="utf-8")
print(summary)

if len(df_inconsistent) > 0:
    print("不一致样本预览:")
    preview_cols = [
        "record_id",
        "gold",
        "llm_pred",
        "rule_pred",
        "final_pred",
        "correction_type",
        "rule_reason",
        "cell_voltage_diff",
        "current_diff",
        "pred_text",
    ]
    print(df_inconsistent[preview_cols].head(20).to_string(index=False))
