from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import ijson


def closest_cycle_cut(cycles: list[dict], target_rows: int, start_index: int = 0) -> int:
    cumulative = sum(cycle["total"] for cycle in cycles[:start_index])
    best_index = start_index
    best_distance = abs(cumulative - target_rows)
    for index in range(start_index, len(cycles)):
        cumulative += cycles[index]["total"]
        distance = abs(cumulative - target_rows)
        if distance <= best_distance:
            best_distance = distance
            best_index = index + 1
        else:
            break
    return best_index


def summarize(cycles: list[dict]) -> dict:
    total = sum(cycle["total"] for cycle in cycles)
    anomaly = sum(cycle["anomaly"] for cycle in cycles)
    normal = total - anomaly
    return {
        "cycles": len(cycles),
        "first_cycle": cycles[0]["cycle_no"] if cycles else None,
        "last_cycle": cycles[-1]["cycle_no"] if cycles else None,
        "first_record_id": cycles[0]["first_record_id"] if cycles else None,
        "last_record_id": cycles[-1]["last_record_id"] if cycles else None,
        "min_timestamp": min(cycle["min_timestamp"] for cycle in cycles) if cycles else None,
        "max_timestamp": max(cycle["max_timestamp"] for cycle in cycles) if cycles else None,
        "total": total,
        "anomaly": anomaly,
        "normal": normal,
        "anomaly_rate": anomaly / total if total else 0.0,
        "balanced_subset_size": 2 * min(anomaly, normal),
    }


def main() -> None:
    project = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()
    source = (
        project
        / "data/processed/record_core_total_confirmed_for_new_error_anomalies.json"
    )

    by_cycle: dict[str, dict] = {}
    with source.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            cycle_no = str(item.get("cycle_no", "missing"))
            record_id = int(item["record_id"])
            timestamp = str(item.get("timestamp", ""))
            label = bool(item["is_anomaly"])
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

    cycles = sorted(by_cycle.values(), key=lambda cycle: cycle["first_record_id"])
    total_rows = sum(cycle["total"] for cycle in cycles)
    train_end = closest_cycle_cut(cycles, round(total_rows * 0.70))
    val_end = closest_cycle_cut(cycles, round(total_rows * 0.85), train_end)

    train_cycles = cycles[:train_end]
    val_cycles = cycles[train_end:val_end]
    test_cycles = cycles[val_end:]

    split_cycles = {
        "train": [cycle["cycle_no"] for cycle in train_cycles],
        "val": [cycle["cycle_no"] for cycle in val_cycles],
        "test": [cycle["cycle_no"] for cycle in test_cycles],
    }
    split_stats = {
        "train": summarize(train_cycles),
        "val": summarize(val_cycles),
        "test": summarize(test_cycles),
    }

    report = {
        "strategy": (
            "Chronological split at complete cycle boundaries, selecting boundaries "
            "closest to 70% and 85% of all records. Train and validation can each "
            "produce balanced subsets using only records inside their own period. "
            "The final test period remains untouched for natural-prevalence metrics; "
            "a balanced test subset may also be sampled only from the same held-out period."
        ),
        "source": str(source),
        "total_cycles": len(cycles),
        "total_rows": total_rows,
        "split_stats": split_stats,
        "split_cycles": split_cycles,
        "cycle_details": cycles,
        "leakage_guards": [
            "No cycle may appear in more than one split.",
            "Splits follow chronological order.",
            "Sampling for balanced train/validation/test subsets happens only after grouping.",
            "The first row of each cycle should not use a previous-row value from another cycle.",
            "BoundarySet remains a separate synthetic stress test and is never used as natural test data.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"split_stats": split_stats, "report_path": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
