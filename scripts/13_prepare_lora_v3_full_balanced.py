import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

IN_PATH = "data/processed/record_core_total_confirmed_for_new_error_anomalies.json"

OUT_DIR = "data/processed/lora_full_balanced_v3"
OUT_TRAIN = os.path.join(OUT_DIR, "train.jsonl")
OUT_VAL = os.path.join(OUT_DIR, "val.jsonl")
OUT_TEST = os.path.join(OUT_DIR, "test.jsonl")

os.makedirs(OUT_DIR, exist_ok=True)

print("读取:", IN_PATH)
df = pd.read_json(IN_PATH)

df["is_anomaly"] = df["is_anomaly"].astype(bool)

# 按记录号排序，重新计算上一行和跳变特征
df = df.sort_values("record_id").reset_index(drop=True)

for col in ["动力电池内部总电压V1", "1号电池单体电压", "动力电池充/放电电流", "1号温度检测点温度"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["prev_total_voltage"] = df["动力电池内部总电压V1"].shift(1)
df["prev_cell_voltage"] = df["1号电池单体电压"].shift(1)
df["prev_current"] = df["动力电池充/放电电流"].shift(1)
df["prev_temperature"] = df["1号温度检测点温度"].shift(1)

df["total_voltage_diff"] = (df["动力电池内部总电压V1"] - df["prev_total_voltage"]).abs()
df["cell_voltage_diff"] = (df["1号电池单体电压"] - df["prev_cell_voltage"]).abs()
df["current_diff"] = (df["动力电池充/放电电流"] - df["prev_current"]).abs()
df["temperature_diff"] = (df["1号温度检测点温度"] - df["prev_temperature"]).abs()

normal_df = df[~df["is_anomaly"]].copy()
anomaly_df = df[df["is_anomaly"]].copy()

print("原始正常样本:", len(normal_df))
print("原始异常样本:", len(anomaly_df))

# 使用全部异常样本，并抽取等量正常样本，构造全量平衡数据集
n_each = len(anomaly_df)
normal_sample = normal_df.sample(n=n_each, random_state=42)
anomaly_sample = anomaly_df.sample(n=n_each, random_state=42)

sample_df = pd.concat([normal_sample, anomaly_sample], axis=0)
sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

def fmt(x, n=4):
    try:
        if pd.isna(x):
            return "缺失"
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)

def make_record(row):
    user = f"""请根据下面的锂电池测试数据判断是否异常，并给出原因。

异常规则：
1. 充电状态下，总电压相邻跳变 > 3V 判为异常；
2. 充电状态下，电流相邻跳变 > 0.5A 判为异常；
3. 充电/放电状态下，单体电压相邻跳变 > 0.05V 判为异常；
4. 充电/放电状态下，温度相邻跳变 > 3℃ 判为异常；
5. 放电状态下，总电压 > 378.2V 判为异常。

当前样本：
记录号：{row.get("record_id")}
时间：{row.get("timestamp")}
工步状态：{row.get("step_status")}
状态编码：{row.get("整车State状态（状态机编码）")}

当前总电压：{fmt(row.get("动力电池内部总电压V1"))} V
上一行总电压：{fmt(row.get("prev_total_voltage"))} V
总电压相邻跳变：{fmt(row.get("total_voltage_diff"))} V

当前单体电压：{fmt(row.get("1号电池单体电压"))} V
上一行单体电压：{fmt(row.get("prev_cell_voltage"))} V
单体电压相邻跳变：{fmt(row.get("cell_voltage_diff"))} V

当前电流：{fmt(row.get("动力电池充/放电电流"))} A
上一行电流：{fmt(row.get("prev_current"))} A
电流相邻跳变：{fmt(row.get("current_diff"))} A

当前温度：{fmt(row.get("1号温度检测点温度"))} ℃
上一行温度：{fmt(row.get("prev_temperature"))} ℃
温度相邻跳变：{fmt(row.get("temperature_diff"))} ℃

请只按照规则输出“正常”或“异常”，并说明触发或未触发的原因。"""

    output = str(row.get("output", ""))
    return {
        "messages": [
            {
                "role": "system",
                "content": "你是电池测试数据异常检测专家，必须严格按照给定阈值和相邻跳变特征判断异常。"
            },
            {
                "role": "user",
                "content": user
            },
            {
                "role": "assistant",
                "content": output
            }
        ],
        "is_anomaly": bool(row["is_anomaly"])
    }

records = [make_record(row) for _, row in sample_df.iterrows()]
labels = [int(x["is_anomaly"]) for x in records]

train_records, temp_records, train_labels, temp_labels = train_test_split(
    records, labels, test_size=0.3, random_state=42, stratify=labels
)

val_records, test_records, val_labels, test_labels = train_test_split(
    temp_records, temp_labels, test_size=1/3, random_state=42, stratify=temp_labels
)

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

write_jsonl(OUT_TRAIN, train_records)
write_jsonl(OUT_VAL, val_records)
write_jsonl(OUT_TEST, test_records)

print("输出完成:")
print(OUT_TRAIN, len(train_records))
print(OUT_VAL, len(val_records))
print(OUT_TEST, len(test_records))
print("训练集异常数:", sum(x["is_anomaly"] for x in train_records))
print("验证集异常数:", sum(x["is_anomaly"] for x in val_records))
print("测试集异常数:", sum(x["is_anomaly"] for x in test_records))

print("\n样例:")
print(json.dumps(train_records[0], ensure_ascii=False, indent=2)[:2000])
