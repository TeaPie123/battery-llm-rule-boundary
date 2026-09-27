from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import ijson


RECORD_RE = re.compile(r"记录号：(\d+)")
SPLITS = ("train", "val", "test")


def load_split_ids(project: Path) -> tuple[dict[str, set[int]], dict[int, str]]:
    split_ids: dict[str, set[int]] = {}
    split_by_id: dict[int, str] = {}
    for split in SPLITS:
        path = project / f"data/processed/lora_full_balanced_v3/{split}.jsonl"
        ids: set[int] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                user_text = item["messages"][1]["content"]
                match = RECORD_RE.search(user_text)
                if not match:
                    raise ValueError(f"Missing record id in {path}:{line_number}")
                record_id = int(match.group(1))
                ids.add(record_id)
                previous = split_by_id.setdefault(record_id, split)
                if previous != split:
                    raise ValueError(
                        f"record_id {record_id} occurs in both {previous} and {split}"
                    )
        split_ids[split] = ids
    return split_ids, split_by_id


def update_range(
    ranges: dict[str, dict[str, object]], split: str, record_id: int, timestamp: str
) -> None:
    current = ranges[split]
    current["min_record_id"] = min(current.get("min_record_id", record_id), record_id)
    current["max_record_id"] = max(current.get("max_record_id", record_id), record_id)
    if timestamp:
        current["min_timestamp"] = min(current.get("min_timestamp", timestamp), timestamp)
        current["max_timestamp"] = max(current.get("max_timestamp", timestamp), timestamp)


def main() -> None:
    project = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()
    source = (
        project
        / "data/processed/record_core_total_confirmed_for_new_error_anomalies.json"
    )

    split_ids, split_by_id = load_split_ids(project)
    split_counts = {split: Counter() for split in (*SPLITS, "unselected")}
    split_ranges: dict[str, dict[str, object]] = defaultdict(dict)
    cycles: dict[str, Counter] = defaultdict(Counter)
    cycle_split_presence: dict[str, set[str]] = defaultdict(set)
    status_counts: dict[str, Counter] = defaultdict(Counter)
    all_ids: list[int] = []

    with source.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            record_id = int(item["record_id"])
            label = bool(item["is_anomaly"])
            cycle = str(item.get("cycle_no", "missing"))
            status = str(item.get("step_status", "missing"))
            timestamp = str(item.get("timestamp", ""))
            split = split_by_id.get(record_id, "unselected")

            all_ids.append(record_id)
            split_counts[split]["total"] += 1
            split_counts[split]["anomaly" if label else "normal"] += 1
            cycles[cycle]["total"] += 1
            cycles[cycle]["anomaly" if label else "normal"] += 1
            cycles[cycle][f"{split}_total"] += 1
            cycles[cycle][f"{split}_anomaly" if label else f"{split}_normal"] += 1
            cycle_split_presence[cycle].add(split)
            status_counts[split][status] += 1
            update_range(split_ranges, split, record_id, timestamp)

    selected_sorted = sorted(split_by_id)
    adjacent_selected_pairs = 0
    adjacent_cross_split_pairs = 0
    adjacent_pair_types = Counter()
    for left, right in zip(selected_sorted, selected_sorted[1:]):
        if right - left != 1:
            continue
        adjacent_selected_pairs += 1
        left_split = split_by_id[left]
        right_split = split_by_id[right]
        adjacent_pair_types[f"{left_split}->{right_split}"] += 1
        if left_split != right_split:
            adjacent_cross_split_pairs += 1

    cycle_overlap = Counter()
    for present in cycle_split_presence.values():
        selected = sorted(present.intersection(SPLITS))
        if len(selected) >= 2:
            cycle_overlap["cycles_in_multiple_selected_splits"] += 1
        if set(SPLITS).issubset(present):
            cycle_overlap["cycles_in_train_val_test"] += 1

    cycle_rows = []
    for cycle, counts in cycles.items():
        row = {"cycle_no": cycle, **counts}
        row["anomaly_rate"] = (
            counts["anomaly"] / counts["total"] if counts["total"] else 0.0
        )
        row["selected_splits"] = sorted(cycle_split_presence[cycle].intersection(SPLITS))
        cycle_rows.append(row)
    cycle_rows.sort(key=lambda row: int(row["cycle_no"]) if row["cycle_no"].isdigit() else 10**9)

    report = {
        "source": str(source),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "split_counts": {key: dict(value) for key, value in split_counts.items()},
        "split_ranges": dict(split_ranges),
        "split_id_counts": {key: len(value) for key, value in split_ids.items()},
        "all_anomalies_used_in_selected_splits": split_counts["unselected"]["anomaly"]
        == 0,
        "unselected_anomaly_count": split_counts["unselected"]["anomaly"],
        "adjacency": {
            "adjacent_selected_pairs": adjacent_selected_pairs,
            "adjacent_cross_split_pairs": adjacent_cross_split_pairs,
            "cross_split_rate_among_adjacent_selected_pairs": (
                adjacent_cross_split_pairs / adjacent_selected_pairs
                if adjacent_selected_pairs
                else 0.0
            ),
            "pair_types": dict(adjacent_pair_types),
        },
        "cycle_summary": {
            "number_of_cycles": len(cycles),
            **dict(cycle_overlap),
            "cycles_with_anomalies": sum(1 for row in cycle_rows if row["anomaly"] > 0),
        },
        "cycles": cycle_rows,
        "status_counts_by_split": {
            key: dict(value.most_common()) for key, value in status_counts.items()
        },
        "conclusion": (
            "The existing random record-level split cannot support an independent "
            "natural-prevalence test because all anomaly records are already assigned "
            "across train/validation/test, and cycle/adjacent-record leakage must be "
            "quantified before creating a grouped temporal split."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(
        {
            "split_counts": report["split_counts"],
            "all_anomalies_used_in_selected_splits": report[
                "all_anomalies_used_in_selected_splits"
            ],
            "adjacency": report["adjacency"],
            "cycle_summary": report["cycle_summary"],
            "report_path": str(out_path),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
