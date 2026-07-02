import json
import pandas as pd
from pathlib import Path

PRED_PATH = Path("outputs/qwen_lora_v3_full_balanced/eval_full_batched_predictions.jsonl")

OUT_DIR = Path("outputs/new_paper_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_WRONG_CSV = OUT_DIR / "v3_wrong_cases.csv"
OUT_METRICS_TXT = OUT_DIR / "v3_baseline_metrics.txt"
OUT_WRONG_MD = OUT_DIR / "v3_wrong_cases_analysis.md"

rows = []

with open(PRED_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

total = len(rows)
valid = sum(r["pred"] is not None for r in rows)
correct = sum(bool(r["correct"]) for r in rows)

tp = sum(r["gold"] is True and r["pred"] is True for r in rows)
fp = sum(r["gold"] is False and r["pred"] is True for r in rows)
tn = sum(r["gold"] is False and r["pred"] is False for r in rows)
fn = sum(r["gold"] is True and r["pred"] is False for r in rows)

acc = correct / valid if valid else 0
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

wrong = [r for r in rows if not r["correct"]]

df_wrong = pd.DataFrame(wrong)
df_wrong.to_csv(OUT_WRONG_CSV, index=False, encoding="utf-8-sig")

metrics_text = f"""v3 full-balanced baseline metrics

Prediction file: {PRED_PATH}

Total samples: {total}
Valid parsed samples: {valid}
Correct samples: {correct}

Accuracy: {acc}
Precision: {precision}
Recall: {recall}
F1: {f1}

TP: {tp}
FP: {fp}
TN: {tn}
FN: {fn}

Wrong cases: {len(wrong)}
Wrong cases csv: {OUT_WRONG_CSV}
"""

OUT_METRICS_TXT.write_text(metrics_text, encoding="utf-8")

md_lines = []
md_lines.append("# v3 wrong cases analysis\n")
md_lines.append("## v3 baseline metrics\n")
md_lines.append("```text\n")
md_lines.append(metrics_text)
md_lines.append("```\n")

md_lines.append("## Wrong cases\n")

for i, r in enumerate(wrong, 1):
    md_lines.append(f"### Wrong case {i}\n")
    md_lines.append(f"- record_id: `{r.get('record_id')}`\n")
    md_lines.append(f"- gold: `{'异常' if r['gold'] else '正常'}`\n")
    md_lines.append(f"- pred: `{'异常' if r['pred'] else '正常' if r['pred'] is False else '无法解析'}`\n")
    md_lines.append("\n**Gold answer:**\n\n")
    md_lines.append(f"```text\n{r.get('gold_text')}\n```\n")
    md_lines.append("\n**Model output:**\n\n")
    md_lines.append(f"```text\n{r.get('pred_text')}\n```\n")

OUT_WRONG_MD.write_text("\n".join(md_lines), encoding="utf-8")

print(metrics_text)

print("输出文件:")
print(OUT_WRONG_CSV)
print(OUT_METRICS_TXT)
print(OUT_WRONG_MD)

print("\n误判样本预览:")
for i, r in enumerate(wrong, 1):
    print("=" * 80)
    print("误判", i)
    print("record_id:", r.get("record_id"))
    print("gold:", "异常" if r["gold"] else "正常")
    print("pred:", "异常" if r["pred"] else "正常" if r["pred"] is False else "无法解析")
    print("gold_text:", r.get("gold_text"))
    print("pred_text:", r.get("pred_text"))
