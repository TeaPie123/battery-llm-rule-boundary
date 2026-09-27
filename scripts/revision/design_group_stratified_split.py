from __future__ import annotations

import json
import sys
from pathlib import Path

import ijson
import numpy as np


CHARGE_STATE = 110
DISCHARGE_STATE = 30


def number(item: dict, key: str) -> float | None:
    value = item.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def corrected_label(item: dict, previous: dict | None) -> bool:
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
    return any(
        (
            state == CHARGE_STATE and total_diff is not None and total_diff > 3.0,
            state == CHARGE_STATE and current_diff is not None and current_diff > 0.5,
            cell_diff is not None and cell_diff > 0.05,
            temperature_diff is not None and temperature_diff > 3.0,
            state == DISCHARGE_STATE
            and total_voltage is not None
            and total_voltage > 378.2,
        )
    )


def stats(cycles: list[dict], mask: np.ndarray) -> dict:
    selected = [cycle for cycle, keep in zip(cycles, mask) if keep]
    total = sum(cycle["total"] for cycle in selected)
    anomaly = sum(cycle["anomaly"] for cycle in selected)
    return {
        "cycles": len(selected),
        "total": total,
        "normal": total - anomaly,
        "anomaly": anomaly,
        "anomaly_rate": anomaly / total if total else 0.0,
        "balanced_subset_size": 2 * min(anomaly, total - anomaly),
        "first_record_id": min((cycle["first_record_id"] for cycle in selected), default=None),
        "last_record_id": max((cycle["last_record_id"] for cycle in selected), default=None),
        "min_timestamp": min((cycle["min_timestamp"] for cycle in selected), default=None),
        "max_timestamp": max((cycle["max_timestamp"] for cycle in selected), default=None),
    }


def optimize_mask(
    rng: np.random.Generator,
    totals: np.ndarray,
    anomalies: np.ndarray,
    allowed: np.ndarray,
    target_total: float,
    target_rate: float,
    probability: float,
    trials: int,
) -> np.ndarray:
    allowed_indices = np.flatnonzero(allowed)
    best_mask = None
    best_score = float("inf")
    for _ in range(trials):
        local = rng.random(len(allowed_indices)) < probability
        if local.sum() < 2:
            continue
        mask = np.zeros(len(totals), dtype=bool)
        mask[allowed_indices[local]] = True
        selected_total = totals[mask].sum()
        selected_anomaly = anomalies[mask].sum()
        selected_rate = selected_anomaly / selected_total
        total_error = abs(selected_total - target_total) / target_total
        rate_error = abs(selected_rate - target_rate) / target_rate
        anomaly_count_target = target_total * target_rate
        anomaly_error = abs(selected_anomaly - anomaly_count_target) / anomaly_count_target
        score = 2.0 * total_error + 4.0 * rate_error + anomaly_error
        if score < best_score:
            best_score = score
            best_mask = mask
    if best_mask is None:
        raise RuntimeError("No valid group mask found")
    return best_mask


def main() -> None:
    project = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()
    source = (
        project
        / "data/processed/record_core_total_confirmed_for_new_error_anomalies.json"
    )

    by_cycle: dict[str, dict] = {}
    previous_by_cycle: dict[str, dict] = {}
    with source.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            cycle_no = str(item.get("cycle_no", "missing"))
            record_id = int(item["record_id"])
            timestamp = str(item.get("timestamp", ""))
            label = corrected_label(item, previous_by_cycle.get(cycle_no))
            cycle = by_cycle.setdefault(
                cycle_no,
                {
                    "cycle_no": cycle_no,
                    "first_record_id": record_id,
                    "last_record_id": record_id,
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "total": 0,
                    "anomaly": 0,
                },
            )
            cycle["first_record_id"] = min(cycle["first_record_id"], record_id)
            cycle["last_record_id"] = max(cycle["last_record_id"], record_id)
            cycle["min_timestamp"] = min(cycle["min_timestamp"], timestamp)
            cycle["max_timestamp"] = max(cycle["max_timestamp"], timestamp)
            cycle["total"] += 1
            cycle["anomaly"] += int(label)
            previous_by_cycle[cycle_no] = item

    cycles = sorted(by_cycle.values(), key=lambda cycle: cycle["first_record_id"])
    totals = np.array([cycle["total"] for cycle in cycles], dtype=np.int64)
    anomalies = np.array([cycle["anomaly"] for cycle in cycles], dtype=np.int64)
    global_total = int(totals.sum())
    global_anomaly = int(anomalies.sum())
    global_rate = global_anomaly / global_total

    rng = np.random.default_rng(20260725)
    all_allowed = np.ones(len(cycles), dtype=bool)
    test_mask = optimize_mask(
        rng,
        totals,
        anomalies,
        all_allowed,
        target_total=global_total * 0.15,
        target_rate=global_rate,
        probability=0.15,
        trials=60000,
    )
    remaining = ~test_mask
    val_mask = optimize_mask(
        rng,
        totals,
        anomalies,
        remaining,
        target_total=global_total * 0.15,
        target_rate=global_rate,
        probability=0.15 / 0.85,
        trials=60000,
    )
    train_mask = ~(test_mask | val_mask)

    split_masks = {"train": train_mask, "val": val_mask, "test": test_mask}
    split_stats = {name: stats(cycles, mask) for name, mask in split_masks.items()}
    split_cycles = {
        name: [cycle["cycle_no"] for cycle, keep in zip(cycles, mask) if keep]
        for name, mask in split_masks.items()
    }

    report = {
        "strategy": (
            "Deterministic cycle-group split optimized to keep train/validation/test "
            "near 70%/15%/15% of records while matching the corrected natural anomaly "
            "prevalence. Labels and previous-row features are recomputed within cycle. "
            "Cycles never overlap. Previous-row features must be recomputed within each "
            "cycle, and the first record of every cycle must have missing previous values."
        ),
        "seed": 20260725,
        "optimization_trials_per_holdout": 60000,
        "global": {
            "cycles": len(cycles),
            "total": global_total,
            "anomaly": global_anomaly,
            "normal": global_total - global_anomaly,
            "anomaly_rate": global_rate,
        },
        "split_stats": split_stats,
        "split_cycles": split_cycles,
        "leakage_guards": [
            "Cycle sets are mutually exclusive.",
            "Balancing is performed only after the cycle split.",
            "Natural test includes every eligible record from held-out test cycles.",
            "Balanced test is sampled only from the same held-out test cycles.",
            "Previous-row features are recomputed within cycle.",
            "BoundarySet is kept separate from all natural-data splits.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"global": report["global"], "split_stats": split_stats, "report_path": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
