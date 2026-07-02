import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

V3_TRAIN = Path("data/processed/lora_full_balanced_v3/train.jsonl")
V3_VAL = Path("data/processed/lora_full_balanced_v3/val.jsonl")
V3_TEST = Path("data/processed/lora_full_balanced_v3/test.jsonl")

BOUNDARY_TRAIN = Path("data/processed/boundary_train/boundary_train.jsonl")
V4_DIR = Path("data/processed/lora_boundary_v4")
V4_TRAIN = V4_DIR / "train.jsonl"
V4_VAL = V4_DIR / "val.jsonl"
V4_TEST = V4_DIR / "test.jsonl"

BOUNDARY_TRAIN.parent.mkdir(parents=True, exist_ok=True)
V4_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM = "你是一个锂电池异常检测专家，需要严格按照给定规则判断样本是否异常。"

def is_anomaly_by_rule(state, total_voltage, total_diff, cell_diff, current_diff, temp_diff):
    reasons = []

    if state == 110:
        if total_diff > 3.0:
            reasons.append(f"充电总电压跳变过大（跳变{total_diff:.4f}V > 阈值3.0V）")
        if current_diff > 0.5:
            reasons.append(f"充电时电流跳变过大（跳变{current_diff:.4f}A > 阈值0.5A）")

    # 当前真实标签逻辑：单体电压和温度跳变作为通用规则，不限制状态
    if cell_diff > 0.05:
        reasons.append(f"1号电池单体电压跳变过大（跳变{cell_diff:.4f}V > 阈值0.05V）")
    if temp_diff > 3.0:
        reasons.append(f"温度跳变过大（跳变{temp_diff:.4f}℃ > 阈值3.0℃）")

    if state == 30:
        if total_voltage > 378.2:
            reasons.append(f"放电总电压超限（当前总电压{total_voltage:.4f}V > 阈值378.2V）")

    if reasons:
        return True, "异常。" + "；".join(reasons)
    return False, "正常。无异常"

def build_user_text(record_id, case_type, state, step_status,
                    total_voltage, prev_total_voltage,
                    cell_voltage, prev_cell_voltage,
                    current, prev_current,
                    temp, prev_temp):

    total_diff = abs(total_voltage - prev_total_voltage)
    cell_diff = abs(cell_voltage - prev_cell_voltage)
    current_diff = abs(current - prev_current)
    temp_diff = abs(temp - prev_temp)

    return f"""请根据下面的锂电池测试数据判断是否异常，并给出原因。

异常规则：
1. 充电状态下，总电压相邻跳变 > 3V 判为异常；
2. 充电状态下，电流相邻跳变 > 0.5A 判为异常；
3. 单体电压相邻跳变 > 0.05V 判为异常；
4. 温度相邻跳变 > 3℃ 判为异常；
5. 放电状态下，总电压 > 378.2V 判为异常。

当前样本：
记录号：{record_id}
边界训练样本类型：{case_type}
时间：boundary_train
工步状态：{step_status}
状态编码：{state}

当前总电压：{total_voltage:.4f} V
上一行总电压：{prev_total_voltage:.4f} V
总电压相邻跳变：{total_diff:.4f} V

当前单体电压：{cell_voltage:.4f} V
上一行单体电压：{prev_cell_voltage:.4f} V
单体电压相邻跳变：{cell_diff:.4f} V

当前电流：{current:.4f} A
上一行电流：{prev_current:.4f} A
电流相邻跳变：{current_diff:.4f} A

当前温度：{temp:.4f} ℃
上一行温度：{prev_temp:.4f} ℃
温度相邻跳变：{temp_diff:.4f} ℃

请只按照规则输出“正常”或“异常”，并说明触发或未触发的原因。"""

def make_item(record_id, case_type, state, step_status,
              total_voltage, prev_total_voltage,
              cell_voltage, prev_cell_voltage,
              current, prev_current,
              temp, prev_temp):

    total_diff = abs(total_voltage - prev_total_voltage)
    cell_diff = abs(cell_voltage - prev_cell_voltage)
    current_diff = abs(current - prev_current)
    temp_diff = abs(temp - prev_temp)

    gold_bool, answer = is_anomaly_by_rule(
        state=state,
        total_voltage=total_voltage,
        total_diff=total_diff,
        cell_diff=cell_diff,
        current_diff=current_diff,
        temp_diff=temp_diff,
    )

    user_text = build_user_text(
        record_id=record_id,
        case_type=case_type,
        state=state,
        step_status=step_status,
        total_voltage=total_voltage,
        prev_total_voltage=prev_total_voltage,
        cell_voltage=cell_voltage,
        prev_cell_voltage=prev_cell_voltage,
        current=current,
        prev_current=prev_current,
        temp=temp,
        prev_temp=prev_temp,
    )

    return {
        "record_id": record_id,
        "case_type": case_type,
        "gold": gold_bool,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ],
    }

items = []
rid = 910000

def add(case_type, state, status, total_voltage, prev_total_voltage,
        cell_voltage, prev_cell_voltage, current, prev_current, temp, prev_temp):
    global rid
    items.append(make_item(
        rid, case_type, state, status,
        total_voltage, prev_total_voltage,
        cell_voltage, prev_cell_voltage,
        current, prev_current,
        temp, prev_temp,
    ))
    rid += 1

# 注意：这里故意不用 BoundarySet 里的完全相同数值，避免测试集泄漏。
current_deltas = [0.08, 0.20, 0.35, 0.48, 0.4955, 0.5005, 0.5055, 0.52, 0.60, 1.20, 10.0, 40.0]
cell_deltas = [0.020, 0.032, 0.040, 0.0485, 0.0495, 0.0505, 0.0520, 0.0560, 0.0700, 0.1200]
temp_deltas = [1.0, 2.5, 2.90, 2.95, 2.9955, 3.0055, 3.02, 3.20, 5.0]
total_jump_deltas = [1.0, 2.5, 2.95, 2.9955, 3.0055, 3.02, 3.20, 5.0]
discharge_voltages = [378.18, 378.195, 378.205, 378.22, 379.0, 400.0]

# 1. 充电电流边界
for base_current in [2.0, 10.0, 30.0, 44.0]:
    for d in current_deltas:
        add(
            "train_charge_current_boundary",
            110,
            "充电",
            3.8, 3.8,
            3.8, 3.8,
            base_current + d, base_current,
            25.0, 25.0,
        )

# 2. 非充电状态下电流跳变适用性
for state, status in [(0, "静置"), (30, "放电")]:
    for base_current in [0.0, 10.0, 30.0]:
        for d in [0.5005, 0.5055, 0.52, 1.2, 10.0, 40.0]:
            add(
                "train_non_charge_current_applicability",
                state,
                status,
                3.8, 3.8,
                3.8, 3.8,
                base_current + d, base_current,
                25.0, 25.0,
            )

# 3. 单体电压边界，三种状态都要覆盖
for state, status in [(0, "静置"), (110, "充电"), (30, "放电")]:
    for base_v in [2.8, 3.0, 3.8, 4.15]:
        for d in cell_deltas:
            add(
                "train_cell_voltage_boundary",
                state,
                status,
                base_v + d, base_v,
                base_v + d, base_v,
                0.0, 0.0,
                25.0, 25.0,
            )

# 4. 温度边界，三种状态都要覆盖
for state, status in [(0, "静置"), (110, "充电"), (30, "放电")]:
    for base_t in [15.0, 25.0, 35.0]:
        for d in temp_deltas:
            add(
                "train_temperature_boundary",
                state,
                status,
                3.8, 3.8,
                3.8, 3.8,
                0.0, 0.0,
                base_t + d, base_t,
            )

# 5. 充电总电压跳变边界
for base_total in [3.8, 100.0, 300.0]:
    for d in total_jump_deltas:
        add(
            "train_charge_total_voltage_jump_boundary",
            110,
            "充电",
            base_total + d, base_total,
            3.8, 3.8,
            10.0, 10.0,
            25.0, 25.0,
        )

# 6. 非充电状态下总电压跳变适用性
for state, status in [(0, "静置"), (30, "放电")]:
    for base_total in [3.8, 300.0]:
        for d in [3.0055, 3.02, 3.20, 5.0]:
            add(
                "train_non_charge_total_voltage_jump_applicability",
                state,
                status,
                base_total + d, base_total,
                3.8, 3.8,
                0.0, 0.0,
                25.0, 25.0,
            )

# 7. 放电总电压超限边界
for v in discharge_voltages:
    add(
        "train_discharge_total_voltage_limit_boundary",
        30,
        "放电",
        v, v,
        3.8, 3.8,
        0.0, 0.0,
        25.0, 25.0,
    )

# 8. 非放电状态下总电压超限适用性
for state, status in [(0, "静置"), (110, "充电")]:
    for v in [378.205, 378.22, 379.0, 400.0]:
        add(
            "train_non_discharge_total_voltage_limit_applicability",
            state,
            status,
            v, v,
            3.8, 3.8,
            0.0, 0.0,
            25.0, 25.0,
        )

# 适度重复边界增强样本，提高训练权重，但不改变测试集
augmented = []
for item in items:
    augmented.append(item)

    # 复制两份，用新 record_id，增强边界训练权重
    for _ in range(2):
        new_item = json.loads(json.dumps(item, ensure_ascii=False))
        new_item["record_id"] = rid
        new_item["messages"][1]["content"] = new_item["messages"][1]["content"].replace(
            f"记录号：{item['record_id']}",
            f"记录号：{rid}"
        )
        augmented.append(new_item)
        rid += 1

items = augmented
random.shuffle(items)

with open(BOUNDARY_TRAIN, "w", encoding="utf-8") as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 合并 v3 train + boundary_train 得到 v4 train
with open(V4_TRAIN, "w", encoding="utf-8") as out:
    for p in [V3_TRAIN, BOUNDARY_TRAIN]:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.write(line)

# val/test 先沿用 v3，避免混淆主测试集
for src, dst in [(V3_VAL, V4_VAL), (V3_TEST, V4_TEST)]:
    with open(src, "r", encoding="utf-8") as fsrc, open(dst, "w", encoding="utf-8") as fdst:
        for line in fsrc:
            if line.strip():
                fdst.write(line)

def count_jsonl(path):
    n = normal = anomaly = 0
    case_counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
                item = json.loads(line)
                msg = item["messages"][-1]["content"]
                if msg.startswith("异常"):
                    anomaly += 1
                else:
                    normal += 1
                if "case_type" in item:
                    case_counter[item["case_type"]] += 1
    return n, normal, anomaly, case_counter

for p in [BOUNDARY_TRAIN, V4_TRAIN, V4_VAL, V4_TEST]:
    n, normal, anomaly, case_counter = count_jsonl(p)
    print("=" * 80)
    print(p)
    print("total:", n)
    print("normal:", normal)
    print("anomaly:", anomaly)
    if case_counter:
        print("case_type counts:")
        for k, v in case_counter.items():
            print(k, v)
