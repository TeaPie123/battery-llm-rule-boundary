import json
from pathlib import Path

OUT_PATH = Path("data/processed/boundary_test/boundary_test.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SYSTEM = "你是一个锂电池异常检测专家，需要严格按照给定规则判断样本是否异常。"

def is_anomaly_by_rule(state, total_voltage, total_diff, cell_diff, current_diff, temp_diff):
    reasons = []

    # 充电状态专用规则
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

    # 放电状态专用规则
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
边界样本类型：{case_type}
时间：boundary_test
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
        "state": state,
        "total_voltage_diff": total_diff,
        "cell_voltage_diff": cell_diff,
        "current_diff": current_diff,
        "temperature_diff": temp_diff,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ],
    }

items = []
rid = 900000

def add(item):
    global rid
    items.append(item)
    rid += 1

# 1. 充电电流跳变边界：严格大于 0.5A 才异常
for d in [0.49, 0.499, 0.50, 0.501, 0.51, 0.07]:
    add(make_item(
        rid, "charge_current_boundary", 110, "充电",
        3.8000, 3.8000,
        3.8000, 3.8000,
        10.0000 + d, 10.0000,
        25.0000, 25.0000,
    ))

# 2. 非充电状态下电流跳变：即使 >0.5A，也不应触发充电电流异常
for state, status in [(0, "静置"), (30, "放电")]:
    for d in [0.501, 0.51, 1.00, 44.48]:
        add(make_item(
            rid, "non_charge_current_applicability", state, status,
            3.8000, 3.8000,
            3.8000, 3.8000,
            10.0000 + d, 10.0000,
            25.0000, 25.0000,
        ))

# 3. 单体电压跳变边界：严格大于 0.05V 才异常，且不限制状态
for state, status in [(0, "静置"), (110, "充电"), (30, "放电")]:
    for d in [0.0308, 0.0343, 0.049, 0.0499, 0.0500, 0.0501, 0.0510, 0.0588]:
        add(make_item(
            rid, "cell_voltage_boundary", state, status,
            3.8000 + d, 3.8000,
            3.8000 + d, 3.8000,
            0.0000, 0.0000,
            25.0000, 25.0000,
        ))

# 4. 温度跳变边界：严格大于 3℃ 才异常，且不限制状态
for state, status in [(0, "静置"), (110, "充电"), (30, "放电")]:
    for d in [2.99, 2.999, 3.000, 3.001, 3.01]:
        add(make_item(
            rid, "temperature_boundary", state, status,
            3.8000, 3.8000,
            3.8000, 3.8000,
            0.0000, 0.0000,
            25.0000 + d, 25.0000,
        ))

# 5. 充电总电压跳变边界：严格大于 3V 才异常
for d in [2.99, 2.999, 3.000, 3.001, 3.01]:
    add(make_item(
        rid, "charge_total_voltage_jump_boundary", 110, "充电",
        300.0000 + d, 300.0000,
        3.8000, 3.8000,
        10.0000, 10.0000,
        25.0000, 25.0000,
    ))

# 6. 非充电状态下总电压跳变：即使 >3V，也不触发充电总电压跳变异常
for state, status in [(0, "静置"), (30, "放电")]:
    for d in [3.001, 3.01]:
        add(make_item(
            rid, "non_charge_total_voltage_jump_applicability", state, status,
            300.0000 + d, 300.0000,
            3.8000, 3.8000,
            0.0000, 0.0000,
            25.0000, 25.0000,
        ))

# 7. 放电总电压超限边界：严格大于 378.2V 才异常
for v in [378.19, 378.199, 378.200, 378.201, 378.210]:
    add(make_item(
        rid, "discharge_total_voltage_limit_boundary", 30, "放电",
        v, v,
        3.8000, 3.8000,
        0.0000, 0.0000,
        25.0000, 25.0000,
    ))

# 8. 非放电状态下总电压超限：不触发放电总电压超限异常
for state, status in [(0, "静置"), (110, "充电")]:
    for v in [378.201, 378.210]:
        add(make_item(
            rid, "non_discharge_total_voltage_limit_applicability", state, status,
            v, v,
            3.8000, 3.8000,
            0.0000, 0.0000,
            25.0000, 25.0000,
        ))

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

normal = sum(not x["gold"] for x in items)
anomaly = sum(x["gold"] for x in items)

print("BoundarySet saved to:", OUT_PATH)
print("total:", len(items))
print("normal:", normal)
print("anomaly:", anomaly)

from collections import Counter
print("case_type counts:")
for k, v in Counter(x["case_type"] for x in items).items():
    print(k, v)
