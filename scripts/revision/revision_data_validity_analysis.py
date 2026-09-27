from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import ijson
import numpy as np


CHARGE_STATE = 110
DISCHARGE_STATE = 30

RULES = {
    "charge_total_jump": {
        "feature": "total_diff",
        "threshold": 3.0,
        "unit": "V",
        "applicability": "state=110 (charge)",
        "windows": {"tight": 0.001, "primary": 0.01, "broad": 0.1},
    },
    "charge_current_jump": {
        "feature": "current_diff",
        "threshold": 0.5,
        "unit": "A",
        "applicability": "state=110 (charge)",
        "windows": {"tight": 0.001, "primary": 0.01, "broad": 0.05},
    },
    "cell_voltage_jump": {
        "feature": "cell_diff",
        "threshold": 0.05,
        "unit": "V",
        "applicability": "all states",
        "windows": {"tight": 0.0001, "primary": 0.001, "broad": 0.005},
    },
    "temperature_jump": {
        "feature": "temperature_diff",
        "threshold": 3.0,
        "unit": "degC",
        "applicability": "all states",
        "windows": {"tight": 0.001, "primary": 0.01, "broad": 0.1},
    },
    "discharge_total_limit": {
        "feature": "total_voltage",
        "threshold": 378.2,
        "unit": "V",
        "applicability": "state=30 (discharge)",
        "windows": {"tight": 0.001, "primary": 0.01, "broad": 0.1},
    },
}

FEATURES = [
    "total_voltage",
    "total_diff",
    "cell_voltage",
    "cell_diff",
    "current",
    "current_diff",
    "temperature",
    "temperature_diff",
]

PREDICTION_FILES = {
    "Base-LoRA": "base_lora_seed42_boundary.jsonl",
    "Pure augmentation": "pure_augmentation_seed42_boundary.jsonl",
    "Boundary weighted": "boundary_weighted_seed42_boundary.jsonl",
}


def as_number(item: dict | None, key: str) -> float | None:
    if not item:
        return None
    value = item.get(key)
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def as_state(item: dict) -> int | None:
    value = as_number(item, "整车State状态（状态机编码）")
    return int(value) if value is not None else None


def difference(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return abs(current - previous)


def state_group(state: int | None) -> str:
    if state == CHARGE_STATE:
        return "charge"
    if state == DISCHARGE_STATE:
        return "discharge"
    return "other_or_idle"


def applicable(rule: str, state: int | None) -> bool:
    if rule in {"charge_total_jump", "charge_current_jump"}:
        return state == CHARGE_STATE
    if rule == "discharge_total_limit":
        return state == DISCHARGE_STATE
    return True


def evaluate_rules(values: dict[str, float | None], state: int | None) -> dict[str, bool]:
    return {
        "charge_total_jump": bool(
            state == CHARGE_STATE
            and values["total_diff"] is not None
            and values["total_diff"] > 3.0
        ),
        "charge_current_jump": bool(
            state == CHARGE_STATE
            and values["current_diff"] is not None
            and values["current_diff"] > 0.5
        ),
        "cell_voltage_jump": bool(
            values["cell_diff"] is not None and values["cell_diff"] > 0.05
        ),
        "temperature_jump": bool(
            values["temperature_diff"] is not None
            and values["temperature_diff"] > 3.0
        ),
        "discharge_total_limit": bool(
            state == DISCHARGE_STATE
            and values["total_voltage"] is not None
            and values["total_voltage"] > 378.2
        ),
    }


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "min": None,
            "p01": None,
            "p05": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    array = np.asarray(values, dtype=float)
    q = np.quantile(array, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "median": float(q[2]),
        "p95": float(q[3]),
        "p99": float(q[4]),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def confusion(gold: list[bool], pred: list[bool]) -> dict[str, float | int | None]:
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    tn = sum((not g) and (not p) for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    total = len(gold)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "samples": total,
        "positive": sum(gold),
        "accuracy": (tp + tn) / total if total else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_prompt_number(prompt: str, label: str, unit: str) -> float | None:
    match = re.search(
        rf"{re.escape(label)}：([+-]?[0-9.]+)\s*{re.escape(unit)}", prompt
    )
    return float(match.group(1)) if match else None


def load_cycle_splits(path: Path) -> dict[str, str]:
    design = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for split, cycles in design["split_cycles"].items():
        for cycle in cycles:
            key = str(cycle)
            if key in result:
                raise ValueError(f"Duplicate cycle in split design: {key}")
            result[key] = split
    return result


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: revision_data_validity_analysis.py PROJECT_DIR")

    project = Path(sys.argv[1]).resolve()
    source = (
        project
        / "data/processed/record_core_total_confirmed_for_new_error_anomalies.json"
    )
    design_path = (
        project
        / "outputs/revision_audit/group_stratified_corrected_labels_design.json"
    )
    boundary_path = project / "data/processed/boundary_test/boundary_test.jsonl"
    eval_dir = project / "outputs/revision_grouped_v1/eval"
    out_dir = project / "outputs/revision_data_validity_v1"
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out_dir}")
    out_dir.mkdir(parents=True)

    cycle_to_split = load_cycle_splits(design_path)
    feature_values: dict[str, list[float]] = defaultdict(list)
    feature_by_split: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    state_counts = Counter()
    state_split_counts: dict[str, Counter] = defaultdict(Counter)
    split_counts = Counter()
    rule_trigger_counts = Counter()
    rule_trigger_by_split: dict[str, Counter] = defaultdict(Counter)
    near_counts: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    near_by_split: dict[str, dict[str, dict[str, Counter]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )
    near_examples: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    label_variants = Counter()
    label_variant_by_split: dict[str, Counter] = defaultdict(Counter)
    overlap_counts = Counter()
    equality = Counter()
    total_minus_cell: list[float] = []
    total_diff_minus_cell_diff: list[float] = []
    previous_by_cycle: dict[str, dict] = {}
    natural_rows = 0

    with source.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            natural_rows += 1
            cycle = str(item.get("cycle_no", "missing"))
            if cycle not in cycle_to_split:
                raise KeyError(f"Cycle absent from split design: {cycle}")
            split = cycle_to_split[cycle]
            previous = previous_by_cycle.get(cycle)
            state = as_state(item)
            total_voltage = as_number(item, "动力电池内部总电压V1")
            cell_voltage = as_number(item, "1号电池单体电压")
            current = as_number(item, "动力电池充/放电电流")
            temperature = as_number(item, "1号温度检测点温度")
            previous_total = as_number(previous, "动力电池内部总电压V1")
            previous_cell = as_number(previous, "1号电池单体电压")
            previous_current = as_number(previous, "动力电池充/放电电流")
            previous_temperature = as_number(previous, "1号温度检测点温度")
            values = {
                "total_voltage": total_voltage,
                "total_diff": difference(total_voltage, previous_total),
                "cell_voltage": cell_voltage,
                "cell_diff": difference(cell_voltage, previous_cell),
                "current": current,
                "current_diff": difference(current, previous_current),
                "temperature": temperature,
                "temperature_diff": difference(temperature, previous_temperature),
            }
            rules = evaluate_rules(values, state)
            original = any(rules.values())
            variants = {
                "original_all_rules": original,
                "remove_charge_total_jump": any(
                    value
                    for name, value in rules.items()
                    if name != "charge_total_jump"
                ),
                "remove_cell_voltage_jump": any(
                    value
                    for name, value in rules.items()
                    if name != "cell_voltage_jump"
                ),
                "remove_discharge_total_limit": any(
                    value
                    for name, value in rules.items()
                    if name != "discharge_total_limit"
                ),
                "remove_both_total_voltage_rules": any(
                    value
                    for name, value in rules.items()
                    if name
                    not in {"charge_total_jump", "discharge_total_limit"}
                ),
            }

            split_counts[split] += 1
            state_name = state_group(state)
            state_counts[state_name] += 1
            state_split_counts[split][state_name] += 1
            for feature, value in values.items():
                if value is not None:
                    feature_values[feature].append(value)
                    feature_by_split[split][feature].append(value)
            for name, triggered in rules.items():
                if triggered:
                    rule_trigger_counts[name] += 1
                    rule_trigger_by_split[split][name] += 1
            for name, label in variants.items():
                if label:
                    label_variants[name] += 1
                    label_variant_by_split[split][name] += 1

            charge_total = rules["charge_total_jump"]
            cell_jump = rules["cell_voltage_jump"]
            overlap_counts["charge_total_trigger"] += int(charge_total)
            overlap_counts["cell_trigger"] += int(cell_jump)
            overlap_counts["both"] += int(charge_total and cell_jump)
            overlap_counts["charge_total_only"] += int(charge_total and not cell_jump)
            overlap_counts["cell_only"] += int(cell_jump and not charge_total)

            if total_voltage is not None and cell_voltage is not None:
                delta = total_voltage - cell_voltage
                total_minus_cell.append(delta)
                equality["current_pairs"] += 1
                equality["current_exact_equal"] += int(delta == 0.0)
            if values["total_diff"] is not None and values["cell_diff"] is not None:
                delta = values["total_diff"] - values["cell_diff"]
                total_diff_minus_cell_diff.append(delta)
                equality["diff_pairs"] += 1
                equality["diff_exact_equal"] += int(delta == 0.0)

            for rule_name, spec in RULES.items():
                if not applicable(rule_name, state):
                    continue
                feature_value = values[spec["feature"]]
                if feature_value is None:
                    continue
                side = (
                    "below_or_equal"
                    if feature_value <= spec["threshold"]
                    else "above"
                )
                distance = abs(feature_value - spec["threshold"])
                for window_name, width in spec["windows"].items():
                    if distance <= width + 1e-12:
                        near_counts[rule_name][window_name]["total"] += 1
                        near_counts[rule_name][window_name][side] += 1
                        near_counts[rule_name][window_name][
                            "label_anomaly" if original else "label_normal"
                        ] += 1
                        near_counts[rule_name][window_name][
                            f"state_{state_name}"
                        ] += 1
                        near_by_split[rule_name][window_name][split]["total"] += 1
                        near_by_split[rule_name][window_name][split][side] += 1
                        near_by_split[rule_name][window_name][split][
                            "label_anomaly" if original else "label_normal"
                        ] += 1
                        near_by_split[rule_name][window_name][split][
                            f"state_{state_name}"
                        ] += 1
                        key = (rule_name, window_name, side)
                        if len(near_examples[key]) < 5:
                            near_examples[key].append(
                                {
                                    "record_id": item.get("record_id"),
                                    "cycle_no": cycle,
                                    "split": split,
                                    "state": state,
                                    "step_status": item.get("step_status"),
                                    "feature_value": feature_value,
                                    "threshold": spec["threshold"],
                                    "distance": distance,
                                    "is_anomaly": original,
                                }
                            )
            previous_by_cycle[cycle] = item

    if natural_rows != sum(split_counts.values()):
        raise AssertionError("Natural row count mismatch")

    boundary_rows = []
    with boundary_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item["messages"][1]["content"]
            values = {
                "total_voltage": parse_prompt_number(
                    prompt, "当前总电压", "V"
                ),
                "total_diff": float(item["total_voltage_diff"]),
                "cell_voltage": parse_prompt_number(
                    prompt, "当前单体电压", "V"
                ),
                "cell_diff": float(item["cell_voltage_diff"]),
                "current": None,
                "current_diff": float(item["current_diff"]),
                "temperature": None,
                "temperature_diff": float(item["temperature_diff"]),
            }
            state = int(item["state"])
            rules = evaluate_rules(values, state)
            variants = {
                "original_all_rules": any(rules.values()),
                "remove_charge_total_jump": any(
                    value
                    for name, value in rules.items()
                    if name != "charge_total_jump"
                ),
                "remove_cell_voltage_jump": any(
                    value
                    for name, value in rules.items()
                    if name != "cell_voltage_jump"
                ),
                "remove_discharge_total_limit": any(
                    value
                    for name, value in rules.items()
                    if name != "discharge_total_limit"
                ),
                "remove_both_total_voltage_rules": any(
                    value
                    for name, value in rules.items()
                    if name not in {"charge_total_jump", "discharge_total_limit"}
                ),
            }
            if variants["original_all_rules"] != bool(item["gold"]):
                raise AssertionError(
                    f"Boundary gold mismatch at record {item['record_id']}"
                )
            boundary_rows.append(
                {
                    "record_id": int(item["record_id"]),
                    "case_type": item["case_type"],
                    "state": state,
                    "state_group": state_group(state),
                    **values,
                    **variants,
                }
            )

    distribution_rows = []
    for source_name, source_values in [
        ("natural_observed_all", feature_values),
        (
            "synthetic_boundary",
            {
                feature: [
                    row[feature]
                    for row in boundary_rows
                    if row.get(feature) is not None
                ]
                for feature in FEATURES
            },
        ),
    ]:
        for feature in FEATURES:
            distribution_rows.append(
                {
                    "source": source_name,
                    "feature": feature,
                    **quantiles(source_values.get(feature, [])),
                }
            )
    write_csv(out_dir / "distribution_summary.csv", distribution_rows)

    state_rows = []
    for split, counter in [("all", state_counts), *sorted(state_split_counts.items())]:
        total = sum(counter.values())
        for name in ["charge", "discharge", "other_or_idle"]:
            state_rows.append(
                {
                    "source": "natural_observed",
                    "split": split,
                    "state_group": name,
                    "count": counter[name],
                    "proportion": counter[name] / total if total else None,
                }
            )
    boundary_states = Counter(row["state_group"] for row in boundary_rows)
    for name in ["charge", "discharge", "other_or_idle"]:
        state_rows.append(
            {
                "source": "synthetic_boundary",
                "split": "boundary",
                "state_group": name,
                "count": boundary_states[name],
                "proportion": boundary_states[name] / len(boundary_rows),
            }
        )
    write_csv(out_dir / "state_distribution.csv", state_rows)

    near_rows = []
    for rule_name, spec in RULES.items():
        for window_name, width in spec["windows"].items():
            counter = near_counts[rule_name][window_name]
            for split in ["all", "train", "val", "test"]:
                current = (
                    counter
                    if split == "all"
                    else near_by_split[rule_name][window_name][split]
                )
                near_rows.append(
                    {
                        "rule": rule_name,
                        "feature": spec["feature"],
                        "threshold": spec["threshold"],
                        "unit": spec["unit"],
                        "applicability": spec["applicability"],
                        "window_name": window_name,
                        "window_half_width": width,
                        "split": split,
                        "count": current["total"],
                        "below_or_equal": current["below_or_equal"],
                        "above": current["above"],
                        "label_normal": current["label_normal"],
                        "label_anomaly": current["label_anomaly"],
                        "state_charge": current["state_charge"],
                        "state_discharge": current["state_discharge"],
                        "state_other_or_idle": current["state_other_or_idle"],
                    }
                )
    write_csv(out_dir / "natural_threshold_coverage.csv", near_rows)

    example_rows = []
    for (rule, window, side), examples in sorted(near_examples.items()):
        for example in examples:
            example_rows.append(
                {"rule": rule, "window_name": window, "side": side, **example}
            )
    write_csv(out_dir / "natural_threshold_examples.csv", example_rows)

    sensitivity_rows = []
    original_count = label_variants["original_all_rules"]
    for variant, count in label_variants.items():
        sensitivity_rows.append(
            {
                "dataset": "natural_observed_all",
                "split": "all",
                "variant": variant,
                "positive_labels": count,
                "changed_vs_original": abs(count - original_count),
                "change_direction": count - original_count,
            }
        )
        for split in ["train", "val", "test"]:
            split_original = label_variant_by_split[split]["original_all_rules"]
            split_count = label_variant_by_split[split][variant]
            sensitivity_rows.append(
                {
                    "dataset": "natural_observed",
                    "split": split,
                    "variant": variant,
                    "positive_labels": split_count,
                    "changed_vs_original": abs(split_count - split_original),
                    "change_direction": split_count - split_original,
                }
            )
    boundary_original = sum(row["original_all_rules"] for row in boundary_rows)
    for variant in [
        "original_all_rules",
        "remove_charge_total_jump",
        "remove_cell_voltage_jump",
        "remove_discharge_total_limit",
        "remove_both_total_voltage_rules",
    ]:
        count = sum(row[variant] for row in boundary_rows)
        changed = sum(
            row[variant] != row["original_all_rules"] for row in boundary_rows
        )
        sensitivity_rows.append(
            {
                "dataset": "synthetic_boundary",
                "split": "boundary",
                "variant": variant,
                "positive_labels": count,
                "changed_vs_original": changed,
                "change_direction": count - boundary_original,
            }
        )
    write_csv(out_dir / "rule_removal_sensitivity.csv", sensitivity_rows)

    redundancy_rows = [
        {
            "metric": "current_value_pairs",
            "value": equality["current_pairs"],
        },
        {
            "metric": "current_exact_equal_pairs",
            "value": equality["current_exact_equal"],
        },
        {
            "metric": "current_exact_equal_rate",
            "value": equality["current_exact_equal"] / equality["current_pairs"],
        },
        {
            "metric": "current_max_abs_difference_v",
            "value": max(map(abs, total_minus_cell), default=None),
        },
        {
            "metric": "diff_pairs",
            "value": equality["diff_pairs"],
        },
        {
            "metric": "diff_exact_equal_pairs",
            "value": equality["diff_exact_equal"],
        },
        {
            "metric": "diff_exact_equal_rate",
            "value": equality["diff_exact_equal"] / equality["diff_pairs"],
        },
        {
            "metric": "diff_max_abs_difference_v",
            "value": max(map(abs, total_diff_minus_cell_diff), default=None),
        },
        *[
            {"metric": f"trigger_overlap_{key}", "value": value}
            for key, value in sorted(overlap_counts.items())
        ],
        *[
            {"metric": f"rule_trigger_{key}", "value": value}
            for key, value in sorted(rule_trigger_counts.items())
        ],
    ]
    write_csv(out_dir / "voltage_redundancy.csv", redundancy_rows)

    corr_features = [
        "total_voltage",
        "total_diff",
        "cell_voltage",
        "cell_diff",
        "current_diff",
        "temperature_diff",
    ]
    # Correlations use complete rows reconstructed from the natural JSONL, whose prompts
    # always expose aligned features except at the first row of each cycle.
    aligned_rows = []
    natural_jsonl_paths = [
        project / "data/processed/revision_grouped_v1/test_natural.jsonl",
        project / "data/processed/revision_grouped_v1/train_balanced.jsonl",
        project / "data/processed/revision_grouped_v1/val_balanced.jsonl",
    ]
    patterns = {
        "total_voltage": r"当前总电压：([+-]?[0-9.]+) V",
        "total_diff": r"总电压相邻跳变：([+-]?[0-9.]+) V",
        "cell_voltage": r"当前单体电压：([+-]?[0-9.]+) V",
        "cell_diff": r"单体电压相邻跳变：([+-]?[0-9.]+) V",
        "current_diff": r"电流相邻跳变：([+-]?[0-9.]+) A",
        "temperature_diff": r"温度相邻跳变：([+-]?[0-9.]+) ℃",
    }
    # The natural test partition is sufficient for a leakage-free natural correlation
    # comparison; balanced train/validation are intentionally not mixed into it.
    with natural_jsonl_paths[0].open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            prompt = item["messages"][1]["content"]
            values = []
            complete = True
            for feature in corr_features:
                match = re.search(patterns[feature], prompt)
                if not match:
                    complete = False
                    break
                values.append(float(match.group(1)))
            if complete:
                aligned_rows.append(values)
    natural_corr = np.corrcoef(np.asarray(aligned_rows, dtype=float), rowvar=False)

    boundary_corr_rows = []
    for row in boundary_rows:
        current = [
            row["total_voltage"],
            row["total_diff"],
            row["cell_voltage"],
            row["cell_diff"],
            row["current_diff"],
            row["temperature_diff"],
        ]
        if all(value is not None for value in current):
            boundary_corr_rows.append(current)
    boundary_corr = np.corrcoef(
        np.asarray(boundary_corr_rows, dtype=float), rowvar=False
    )
    correlation_rows = []
    for source_name, matrix in [
        ("natural_test_complete_rows", natural_corr),
        ("synthetic_boundary", boundary_corr),
    ]:
        for i, row_feature in enumerate(corr_features):
            for j, col_feature in enumerate(corr_features):
                value = float(matrix[i, j])
                correlation_rows.append(
                    {
                        "source": source_name,
                        "row_feature": row_feature,
                        "column_feature": col_feature,
                        "pearson_r": value if math.isfinite(value) else None,
                    }
                )
    write_csv(out_dir / "feature_correlations.csv", correlation_rows)

    prediction_rows = []
    for model_name, filename in PREDICTION_FILES.items():
        predictions = {}
        with (eval_dir / filename).open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                predictions[int(item["record_id"])] = bool(item["pred"])
        if set(predictions) != {row["record_id"] for row in boundary_rows}:
            raise ValueError(f"Boundary prediction IDs mismatch for {model_name}")
        pred = [predictions[row["record_id"]] for row in boundary_rows]
        for variant in [
            "original_all_rules",
            "remove_charge_total_jump",
            "remove_cell_voltage_jump",
            "remove_discharge_total_limit",
            "remove_both_total_voltage_rules",
        ]:
            gold = [bool(row[variant]) for row in boundary_rows]
            prediction_rows.append(
                {"model": model_name, "gold_variant": variant, **confusion(gold, pred)}
            )
    write_csv(out_dir / "boundary_model_rescoring.csv", prediction_rows)

    protocol = {
        "version": "revision_data_validity_v1",
        "sources": {
            "natural_observed": str(source),
            "cycle_split_design": str(design_path),
            "synthetic_boundary": str(boundary_path),
            "boundary_predictions": {
                model: str(eval_dir / filename)
                for model, filename in PREDICTION_FILES.items()
            },
        },
        "terminology": {
            "natural_observed": (
                "Existing laboratory/test-bench observations from the original source; "
                "not external industrial boundary validation."
            ),
            "synthetic_boundary": (
                "Rule-generated synthetic stress-test cases; not observed field events."
            ),
        },
        "window_rationale": (
            "The primary half-width matches the widest near-threshold perturbation "
            "used in BoundarySet (0.01 in the rule's unit, or 0.001 V for the "
            "0.05 V cell-jump rule). Tight and broad bands are sensitivity checks."
        ),
        "rules": RULES,
        "counts": {
            "natural_observed_rows": natural_rows,
            "synthetic_boundary_rows": len(boundary_rows),
            "natural_split_rows": dict(split_counts),
        },
    }
    (out_dir / "analysis_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    primary_coverage = {
        rule: near_counts[rule]["primary"]["total"] for rule in RULES
    }
    natural_change_remove_total = (
        label_variants["original_all_rules"]
        - label_variants["remove_both_total_voltage_rules"]
    )
    boundary_change_remove_total = sum(
        row["original_all_rules"] != row["remove_both_total_voltage_rules"]
        for row in boundary_rows
    )
    report = f"""# Data-validity analysis for revision items 3-5

## Scope and terminology

- Natural observed data: {natural_rows:,} existing test records, partitioned by cycle into {dict(split_counts)}.
- Synthetic BoundarySet: {len(boundary_rows)} rule-generated stress-test cases.
- The natural observations are not external industrial boundary validation.

## Primary near-threshold coverage

The primary bands follow the BoundarySet perturbation scale. Counts across all observed records:

{json.dumps(primary_coverage, ensure_ascii=False, indent=2)}

See `natural_threshold_coverage.csv` for tight/primary/broad bands, sides of the threshold,
labels, states, and train/validation/test breakdowns.

## Voltage-field redundancy

- Current total-voltage/cell-voltage exact equality:
  {equality["current_exact_equal"]:,}/{equality["current_pairs"]:,}
  ({equality["current_exact_equal"] / equality["current_pairs"]:.6%}).
- Adjacent-difference exact equality:
  {equality["diff_exact_equal"]:,}/{equality["diff_pairs"]:,}
  ({equality["diff_exact_equal"] / equality["diff_pairs"]:.6%}).
- Charge-total-jump only triggers: {overlap_counts["charge_total_only"]:,}.
- Charge-total and cell-jump simultaneous triggers: {overlap_counts["both"]:,}.
- Natural labels changed after removing both total-voltage rules:
  {natural_change_remove_total:,}.
- BoundarySet gold labels changed after removing both total-voltage rules:
  {boundary_change_remove_total:,}/{len(boundary_rows)}.

## Interpretation guardrails

1. Near-threshold observed records support only in-source coverage analysis, not industrial deployment claims.
2. A zero count means the corresponding rule boundary is covered only synthetically in the present evidence.
3. Because total voltage and cell voltage are identical in the observed source mapping, independent PACK-level
   and cell-level physical conclusions are not identifiable.
4. Boundary model rescoring changes the gold definition only; it does not retrain or alter model predictions.
"""
    (out_dir / "analysis_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "version": "revision_data_validity_v1",
        "files": {},
    }
    for path in sorted(out_dir.iterdir()):
        if path.name == "provenance_manifest.json" or not path.is_file():
            continue
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    (out_dir / "provenance_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(primary_coverage, ensure_ascii=False, indent=2))
    print(f"OUTPUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
