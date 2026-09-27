from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import ijson


CHARGE_STATE = 110
DISCHARGE_STATE = 30
SEED = 20260725


def number(item: dict, key: str) -> float | None:
    value = item.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: float | None, digits: int = 4) -> str:
    return "缺失" if value is None else f"{value:.{digits}f}"


def evaluate(item: dict, previous: dict | None) -> tuple[dict[str, bool], dict[str, float | None]]:
    state_value = number(item, "整车State状态（状态机编码）")
    state = int(state_value) if state_value is not None else None
    total_voltage = number(item, "动力电池内部总电压V1")
    cell_voltage = number(item, "1号电池单体电压")
    current = number(item, "动力电池充/放电电流")
    temperature = number(item, "1号温度检测点温度")
    previous_total = number(previous, "动力电池内部总电压V1") if previous else None
    previous_cell = number(previous, "1号电池单体电压") if previous else None
    previous_current = number(previous, "动力电池充/放电电流") if previous else None
    previous_temperature = number(previous, "1号温度检测点温度") if previous else None

    total_diff = (
        abs(total_voltage - previous_total)
        if total_voltage is not None and previous_total is not None
        else None
    )
    cell_diff = (
        abs(cell_voltage - previous_cell)
        if cell_voltage is not None and previous_cell is not None
        else None
    )
    current_diff = (
        abs(current - previous_current)
        if current is not None and previous_current is not None
        else None
    )
    temperature_diff = (
        abs(temperature - previous_temperature)
        if temperature is not None and previous_temperature is not None
        else None
    )
    rules = {
        "charge_total_jump": bool(
            state == CHARGE_STATE and total_diff is not None and total_diff > 3.0
        ),
        "charge_current_jump": bool(
            state == CHARGE_STATE and current_diff is not None and current_diff > 0.5
        ),
        "cell_voltage_jump": bool(cell_diff is not None and cell_diff > 0.05),
        "temperature_jump": bool(
            temperature_diff is not None and temperature_diff > 3.0
        ),
        "discharge_total_limit": bool(
            state == DISCHARGE_STATE
            and total_voltage is not None
            and total_voltage > 378.2
        ),
    }
    values = {
        "state": state,
        "total_voltage": total_voltage,
        "previous_total": previous_total,
        "total_diff": total_diff,
        "cell_voltage": cell_voltage,
        "previous_cell": previous_cell,
        "cell_diff": cell_diff,
        "current": current,
        "previous_current": previous_current,
        "current_diff": current_diff,
        "temperature": temperature,
        "previous_temperature": previous_temperature,
        "temperature_diff": temperature_diff,
    }
    return rules, values


def output_text(rules: dict[str, bool], values: dict[str, float | None]) -> str:
    reasons = []
    if rules["charge_total_jump"]:
        reasons.append(f"充电时总电压跳变{values['total_diff']:.4f}V > 3.0V")
    if rules["charge_current_jump"]:
        reasons.append(f"充电时电流跳变{values['current_diff']:.4f}A > 0.5A")
    if rules["cell_voltage_jump"]:
        reasons.append(f"单体电压跳变{values['cell_diff']:.4f}V > 0.05V")
    if rules["temperature_jump"]:
        reasons.append(f"温度跳变{values['temperature_diff']:.4f}℃ > 3.0℃")
    if rules["discharge_total_limit"]:
        reasons.append(f"放电时总电压{values['total_voltage']:.4f}V > 378.2V")
    return f"异常。{'; '.join(reasons)}" if reasons else "正常。无异常"


def make_record(item: dict, previous: dict | None, split: str) -> dict:
    rules, values = evaluate(item, previous)
    is_anomaly = any(rules.values())
    user = f"""请根据下面的锂电池测试数据判断是否异常，并给出原因。

异常规则：
1. 充电状态下，总电压相邻跳变 > 3V 判为异常；
2. 充电状态下，电流相邻跳变 > 0.5A 判为异常；
3. 所有运行状态下，单体电压相邻跳变 > 0.05V 判为异常；
4. 所有运行状态下，温度相邻跳变 > 3℃ 判为异常；
5. 放电状态下，总电压 > 378.2V 判为异常。

当前样本：
记录号：{item.get("record_id")}
循环号：{item.get("cycle_no")}
时间：{item.get("timestamp")}
工步状态：{item.get("step_status")}
状态编码：{item.get("整车State状态（状态机编码）")}

当前总电压：{fmt(values["total_voltage"])} V
本循环上一行总电压：{fmt(values["previous_total"])} V
总电压相邻跳变：{fmt(values["total_diff"])} V

当前单体电压：{fmt(values["cell_voltage"])} V
本循环上一行单体电压：{fmt(values["previous_cell"])} V
单体电压相邻跳变：{fmt(values["cell_diff"])} V

当前电流：{fmt(values["current"])} A
本循环上一行电流：{fmt(values["previous_current"])} A
电流相邻跳变：{fmt(values["current_diff"])} A

当前温度：{fmt(values["temperature"])} ℃
本循环上一行温度：{fmt(values["previous_temperature"])} ℃
温度相邻跳变：{fmt(values["temperature_diff"])} ℃

请只按照规则输出“正常”或“异常”，并说明触发或未触发的原因。"""
    return {
        "messages": [
            {
                "role": "system",
                "content": "你是电池测试数据异常检测专家，必须严格按照给定阈值、运行状态和本循环相邻跳变特征判断异常。",
            },
            {"role": "user", "content": user},
            {"role": "assistant", "content": output_text(rules, values)},
        ],
        "is_anomaly": is_anomaly,
        "record_id": int(item["record_id"]),
        "cycle_no": str(item.get("cycle_no", "missing")),
        "split": split,
        "rule_triggers": [name for name, triggered in rules.items() if triggered],
    }


def reservoir_add(
    reservoir: list[dict],
    record: dict,
    seen: int,
    target: int,
    rng: random.Random,
) -> None:
    if len(reservoir) < target:
        reservoir.append(record)
        return
    replacement = rng.randrange(seen)
    if replacement < target:
        reservoir[replacement] = record


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    project = Path(sys.argv[1]).resolve()
    design_path = Path(sys.argv[2]).resolve()
    final_dir = Path(sys.argv[3]).resolve()
    temp_dir = final_dir.with_name(final_dir.name + ".tmp")
    if final_dir.exists() or temp_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {final_dir} or {temp_dir}"
        )

    with design_path.open("r", encoding="utf-8") as handle:
        design = json.load(handle)
    cycle_to_split: dict[str, str] = {}
    for split, cycles in design["split_cycles"].items():
        for cycle in cycles:
            cycle_key = str(cycle)
            if cycle_key in cycle_to_split:
                raise ValueError(f"Cycle {cycle_key} appears in multiple splits")
            cycle_to_split[cycle_key] = split

    expected_anomalies = {
        split: int(stats["anomaly"])
        for split, stats in design["split_stats"].items()
    }
    source = (
        project
        / "data/processed/record_core_total_confirmed_for_new_error_anomalies.json"
    )
    temp_dir.mkdir(parents=True)
    natural_test_path = temp_dir / "test_natural.jsonl"
    natural_test_handle = natural_test_path.open("w", encoding="utf-8")

    rngs = {
        "train": random.Random(SEED + 1),
        "val": random.Random(SEED + 2),
        "test": random.Random(SEED + 3),
    }
    anomaly_records: dict[str, list[dict]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    normal_reservoirs: dict[str, list[dict]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    normal_seen = Counter()
    counts = {split: Counter() for split in ("train", "val", "test")}
    previous_by_cycle: dict[str, dict] = {}

    try:
        with source.open("rb") as handle:
            for item in ijson.items(handle, "item"):
                cycle = str(item.get("cycle_no", "missing"))
                split = cycle_to_split[cycle]
                previous = previous_by_cycle.get(cycle)
                record = make_record(item, previous, split)
                label = bool(record["is_anomaly"])
                counts[split]["total"] += 1
                counts[split]["anomaly" if label else "normal"] += 1
                if split == "test":
                    natural_test_handle.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                if label:
                    anomaly_records[split].append(record)
                else:
                    normal_seen[split] += 1
                    reservoir_add(
                        normal_reservoirs[split],
                        record,
                        normal_seen[split],
                        expected_anomalies[split],
                        rngs[split],
                    )
                previous_by_cycle[cycle] = item
    finally:
        natural_test_handle.close()

    for split in ("train", "val", "test"):
        if len(anomaly_records[split]) != expected_anomalies[split]:
            raise ValueError(
                f"{split} anomaly mismatch: {len(anomaly_records[split])} "
                f"!= {expected_anomalies[split]}"
            )
        if len(normal_reservoirs[split]) != expected_anomalies[split]:
            raise ValueError(f"{split} normal reservoir is incomplete")
        balanced = anomaly_records[split] + normal_reservoirs[split]
        rngs[split].shuffle(balanced)
        write_jsonl(temp_dir / f"{split}_balanced.jsonl", balanced)

    manifest = {
        "version": "revision_grouped_v1",
        "seed": SEED,
        "source": str(source),
        "design": str(design_path),
        "rules": {
            "charge_total_jump_v": 3.0,
            "charge_current_jump_a": 0.5,
            "cell_voltage_jump_v_all_states": 0.05,
            "temperature_jump_c_all_states": 3.0,
            "discharge_total_limit_v": 378.2,
            "charge_state": CHARGE_STATE,
            "discharge_state": DISCHARGE_STATE,
            "previous_row_scope": "within_cycle",
        },
        "counts": {split: dict(value) for split, value in counts.items()},
        "files": {},
    }
    for path in sorted(temp_dir.glob("*.jsonl")):
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    with (temp_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    temp_dir.rename(final_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
