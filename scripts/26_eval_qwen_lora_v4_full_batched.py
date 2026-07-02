import json
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "models/Qwen/Qwen2___5-0___5B-Instruct"
LORA_DIR = "outputs/qwen_lora_v4_boundary"
TEST_PATH = "data/processed/lora_boundary_v4/test_clean.jsonl"
OUT_PATH = "outputs/qwen_lora_v4_boundary/eval_full_batched_predictions.jsonl"

BATCH_SIZE = 16
MAX_NEW_TOKENS = 80

print("基础模型:", BASE_MODEL)
print("LoRA adapter:", LORA_DIR)
print("测试集:", TEST_PATH)
print("输出文件:", OUT_PATH)
print("cuda available:", torch.cuda.is_available())

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, LORA_DIR)

if torch.cuda.is_available():
    model = model.cuda()

model.eval()

records = []
with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print("测试样本数:", len(records))

def build_prompt(item):
    return tokenizer.apply_chat_template(
        item["messages"][:-1],
        tokenize=False,
        add_generation_prompt=True
    )

def parse_label(text):
    t = text.strip()
    if t.startswith("异常"):
        return True
    if t.startswith("正常"):
        return False

    head = t[:40]
    if "异常" in head and "正常" not in head:
        return True
    if "正常" in head:
        return False
    return None

def get_record_id(item):
    user_text = item["messages"][1]["content"]
    m = re.search(r"记录号：(\d+)", user_text)
    return m.group(1) if m else None

correct = 0
valid = 0
tp = fp = tn = fn = 0

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

with open(OUT_PATH, "w", encoding="utf-8") as fout:
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        prompts = [build_prompt(x) for x in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        input_len = inputs["input_ids"].shape[1]

        for j, item in enumerate(batch):
            pred_text = tokenizer.decode(
                outputs[j][input_len:],
                skip_special_tokens=True
            ).strip()

            gold = bool(item["is_anomaly"])
            pred = parse_label(pred_text)

            if pred is not None:
                valid += 1
                if pred == gold:
                    correct += 1

                if pred is True and gold is True:
                    tp += 1
                elif pred is True and gold is False:
                    fp += 1
                elif pred is False and gold is False:
                    tn += 1
                elif pred is False and gold is True:
                    fn += 1

            if start + j < 10:
                print("=" * 80)
                print("样本", start + j + 1)
                print("真实标签:", "异常" if gold else "正常")
                print("标准答案:", item["messages"][-1]["content"])
                print("模型输出:", pred_text)
                print("解析标签:", "异常" if pred is True else "正常" if pred is False else "无法解析")

            fout.write(json.dumps({
                "record_id": get_record_id(item),
                "gold": gold,
                "pred": pred,
                "correct": pred == gold if pred is not None else False,
                "gold_text": item["messages"][-1]["content"],
                "pred_text": pred_text
            }, ensure_ascii=False) + "\n")

        done = min(start + BATCH_SIZE, len(records))
        if done % 100 == 0 or done == len(records):
            print(f"进度: {done}/{len(records)}")

print("=" * 80)
print("v4完整测试集可解析数量:", valid)
print("v4完整测试集正确数量:", correct)

if valid > 0:
    acc = correct / valid
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("v4完整测试集准确率:", acc)
    print("TP:", tp)
    print("FP:", fp)
    print("TN:", tn)
    print("FN:", fn)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
