import json
import re
from pathlib import Path

DIR = Path("data/processed/lora_full_balanced_v3")

def load_ids(path):
    ids = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            user_text = item["messages"][1]["content"]
            m = re.search(r"记录号：(\d+)", user_text)
            if m:
                ids.append(m.group(1))
            labels.append(bool(item["is_anomaly"]))
    return set(ids), labels

train_ids, train_labels = load_ids(DIR / "train.jsonl")
val_ids, val_labels = load_ids(DIR / "val.jsonl")
test_ids, test_labels = load_ids(DIR / "test.jsonl")

print("train ids:", len(train_ids))
print("val ids:", len(val_ids))
print("test ids:", len(test_ids))

print("train/val overlap:", len(train_ids & val_ids))
print("train/test overlap:", len(train_ids & test_ids))
print("val/test overlap:", len(val_ids & test_ids))

print("train anomalies:", sum(train_labels), "normal:", len(train_labels) - sum(train_labels))
print("val anomalies:", sum(val_labels), "normal:", len(val_labels) - sum(val_labels))
print("test anomalies:", sum(test_labels), "normal:", len(test_labels) - sum(test_labels))
