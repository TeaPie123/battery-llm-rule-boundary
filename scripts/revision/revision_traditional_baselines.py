from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import ijson
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


SEED = 42
CHARGE_STATE = 110
DISCHARGE_STATE = 30

FEATURE_NAMES = [
    "state_charge",
    "state_discharge",
    "state_other_or_idle",
    "total_voltage",
    "previous_total_voltage",
    "total_voltage_diff",
    "cell_voltage",
    "previous_cell_voltage",
    "cell_voltage_diff",
    "current",
    "previous_current",
    "current_diff",
    "temperature",
    "previous_temperature",
    "temperature_diff",
    "has_previous_in_cycle",
]

DATASET_ORDER = ["balanced", "natural", "boundary"]


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


def difference(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return abs(current - previous)


def state_value(item: dict) -> int | None:
    value = as_number(item, "整车State状态（状态机编码）")
    return int(value) if value is not None else None


def rules_and_label(
    state: int | None,
    total_voltage: float | None,
    total_diff: float | None,
    cell_diff: float | None,
    current_diff: float | None,
    temperature_diff: float | None,
) -> tuple[dict[str, bool], bool]:
    rules = {
        "charge_total_jump": bool(
            state == CHARGE_STATE
            and total_diff is not None
            and total_diff > 3.0
        ),
        "charge_current_jump": bool(
            state == CHARGE_STATE
            and current_diff is not None
            and current_diff > 0.5
        ),
        "cell_voltage_jump": bool(
            cell_diff is not None and cell_diff > 0.05
        ),
        "temperature_jump": bool(
            temperature_diff is not None and temperature_diff > 3.0
        ),
        "discharge_total_limit": bool(
            state == DISCHARGE_STATE
            and total_voltage is not None
            and total_voltage > 378.2
        ),
    }
    return rules, any(rules.values())


def feature_vector(
    state: int | None,
    total_voltage: float | None,
    previous_total: float | None,
    cell_voltage: float | None,
    previous_cell: float | None,
    current: float | None,
    previous_current: float | None,
    temperature: float | None,
    previous_temperature: float | None,
) -> tuple[list[float], dict[str, float | None]]:
    total_diff = difference(total_voltage, previous_total)
    cell_diff = difference(cell_voltage, previous_cell)
    current_diff = difference(current, previous_current)
    temperature_diff = difference(temperature, previous_temperature)
    has_previous = all(
        value is not None
        for value in [
            previous_total,
            previous_cell,
            previous_current,
            previous_temperature,
        ]
    )

    def current_or_zero(value: float | None) -> float:
        return 0.0 if value is None else value

    def previous_or_current(
        previous_value: float | None, current_value: float | None
    ) -> float:
        if previous_value is not None:
            return previous_value
        return current_or_zero(current_value)

    values = {
        "total_voltage": total_voltage,
        "total_diff": total_diff,
        "cell_voltage": cell_voltage,
        "cell_diff": cell_diff,
        "current": current,
        "current_diff": current_diff,
        "temperature": temperature,
        "temperature_diff": temperature_diff,
    }
    vector = [
        float(state == CHARGE_STATE),
        float(state == DISCHARGE_STATE),
        float(state not in {CHARGE_STATE, DISCHARGE_STATE}),
        current_or_zero(total_voltage),
        previous_or_current(previous_total, total_voltage),
        current_or_zero(total_diff),
        current_or_zero(cell_voltage),
        previous_or_current(previous_cell, cell_voltage),
        current_or_zero(cell_diff),
        current_or_zero(current),
        previous_or_current(previous_current, current),
        current_or_zero(current_diff),
        current_or_zero(temperature),
        previous_or_current(previous_temperature, temperature),
        current_or_zero(temperature_diff),
        float(has_previous),
    ]
    return vector, values


def load_jsonl_id_labels(path: Path) -> dict[int, bool]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            record_id = int(item["record_id"])
            if record_id in result:
                raise ValueError(f"Duplicate record ID in {path}: {record_id}")
            result[record_id] = bool(item["is_anomaly"])
    return result


def load_cycle_splits(path: Path) -> dict[str, str]:
    design = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for split, cycles in design["split_cycles"].items():
        for cycle in cycles:
            cycle = str(cycle)
            if cycle in result:
                raise ValueError(f"Cycle appears in multiple splits: {cycle}")
            result[cycle] = split
    return result


def parse_prompt_number(prompt: str, label: str, unit: str) -> float | None:
    pattern = rf"{re.escape(label)}：(?:缺失|([+-]?[0-9.]+))\s*{re.escape(unit)}"
    match = re.search(pattern, prompt)
    if not match or match.group(1) is None:
        return None
    return float(match.group(1))


def parse_prompt_state(prompt: str) -> int | None:
    match = re.search(r"状态编码：([+-]?\d+)", prompt)
    return int(match.group(1)) if match else None


def load_boundary(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = []
    vectors = []
    labels = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item["messages"][1]["content"]
            state = int(item.get("state", parse_prompt_state(prompt)))
            total_voltage = parse_prompt_number(prompt, "当前总电压", "V")
            previous_total = parse_prompt_number(prompt, "上一行总电压", "V")
            cell_voltage = parse_prompt_number(prompt, "当前单体电压", "V")
            previous_cell = parse_prompt_number(prompt, "上一行单体电压", "V")
            current = parse_prompt_number(prompt, "当前电流", "A")
            previous_current = parse_prompt_number(prompt, "上一行电流", "A")
            temperature = parse_prompt_number(prompt, "当前温度", "℃")
            previous_temperature = parse_prompt_number(prompt, "上一行温度", "℃")
            vector, values = feature_vector(
                state,
                total_voltage,
                previous_total,
                cell_voltage,
                previous_cell,
                current,
                previous_current,
                temperature,
                previous_temperature,
            )
            # BoundarySet labels were generated from the builder's unrounded
            # top-level differences. The prompt displays four decimals, which can
            # turn 0.04999999999999982 into 0.0500 and create a false strict-
            # inequality mismatch if differences are recomputed from prompt text.
            values["total_diff"] = float(item["total_voltage_diff"])
            values["cell_diff"] = float(item["cell_voltage_diff"])
            values["current_diff"] = float(item["current_diff"])
            values["temperature_diff"] = float(item["temperature_diff"])
            vector[5] = values["total_diff"]
            vector[8] = values["cell_diff"]
            vector[11] = values["current_diff"]
            vector[14] = values["temperature_diff"]
            _, label = rules_and_label(
                state,
                values["total_voltage"],
                values["total_diff"],
                values["cell_diff"],
                values["current_diff"],
                values["temperature_diff"],
            )
            if label != bool(item["gold"]):
                raise ValueError(
                    f"Boundary label mismatch at record {item['record_id']}"
                )
            vectors.append(vector)
            labels.append(label)
            rows.append(
                {
                    "record_id": int(item["record_id"]),
                    "cycle_no": None,
                    "case_type": item.get("case_type"),
                    "gold": label,
                }
            )
    return (
        np.asarray(vectors, dtype=np.float64),
        np.asarray(labels, dtype=bool),
        rows,
    )


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        raise ValueError("Cannot tune threshold without score variation")
    numerator = 2.0 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    best = np.flatnonzero(f1 == np.nanmax(f1))
    # Conservative tie-break: prefer the highest threshold, i.e. fewer false positives.
    index = int(best[-1])
    return {
        "threshold": float(thresholds[index]),
        "validation_f1": float(f1[index]),
        "validation_precision": float(precision[index]),
        "validation_recall": float(recall[index]),
        "tie_break": "highest threshold among equal maximum validation F1",
    }


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[False, True]
    ).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "samples": int(y_true.size),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "fnr": float(fn / (fn + tp)) if fn + tp else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "auroc": (
            float(roc_auc_score(y_true, scores))
            if np.unique(y_true).size == 2
            else None
        ),
        "auprc": (
            float(average_precision_score(y_true, scores))
            if np.unique(y_true).size == 2
            else None
        ),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def write_predictions(
    path: Path,
    row_metadata: list[dict],
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for metadata, gold, score in zip(row_metadata, y_true, scores):
            output = dict(metadata)
            output.update(
                {
                    "gold": bool(gold),
                    "pred": bool(score >= threshold),
                    "anomaly_score": float(score),
                }
            )
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_memory_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: revision_traditional_baselines.py PROJECT_DIR")
    project = Path(sys.argv[1]).resolve()
    source = (
        project
        / "data/processed/record_core_total_confirmed_for_new_error_anomalies.json"
    )
    grouped_dir = project / "data/processed/revision_grouped_v1"
    design_path = (
        project
        / "outputs/revision_audit/group_stratified_corrected_labels_design.json"
    )
    boundary_path = project / "data/processed/boundary_test/boundary_test.jsonl"
    final_output_dir = project / "outputs/revision_traditional_baselines_v1"
    output_dir = project / "outputs/revision_traditional_baselines_v1.tmp"
    if final_output_dir.exists() or output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {final_output_dir} or {output_dir}"
        )
    output_dir.mkdir(parents=True)

    balanced_labels = {
        split: load_jsonl_id_labels(grouped_dir / f"{split}_balanced.jsonl")
        for split in ("train", "val", "test")
    }
    id_to_target = {}
    for split, labels in balanced_labels.items():
        for record_id in labels:
            if record_id in id_to_target:
                raise ValueError(f"Balanced record in multiple splits: {record_id}")
            id_to_target[record_id] = split

    cycle_to_split = load_cycle_splits(design_path)
    features = {split: [] for split in ("train", "val", "test_balanced", "test_natural")}
    labels = {split: [] for split in features}
    metadata = {split: [] for split in features}
    previous_by_cycle = {}
    source_count = 0
    rule_counts = Counter()

    extraction_started = time.perf_counter()
    with source.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            source_count += 1
            record_id = int(item["record_id"])
            cycle = str(item.get("cycle_no", "missing"))
            split = cycle_to_split[cycle]
            previous = previous_by_cycle.get(cycle)
            state = state_value(item)
            vector, values = feature_vector(
                state,
                as_number(item, "动力电池内部总电压V1"),
                as_number(previous, "动力电池内部总电压V1"),
                as_number(item, "1号电池单体电压"),
                as_number(previous, "1号电池单体电压"),
                as_number(item, "动力电池充/放电电流"),
                as_number(previous, "动力电池充/放电电流"),
                as_number(item, "1号温度检测点温度"),
                as_number(previous, "1号温度检测点温度"),
            )
            rules, label = rules_and_label(
                state,
                values["total_voltage"],
                values["total_diff"],
                values["cell_diff"],
                values["current_diff"],
                values["temperature_diff"],
            )
            for name, triggered in rules.items():
                rule_counts[name] += int(triggered)

            if record_id in id_to_target:
                target_split = id_to_target[record_id]
                if target_split != split:
                    raise ValueError(
                        f"Record/cycle split mismatch at {record_id}: "
                        f"{target_split} vs {split}"
                    )
                official_label = balanced_labels[target_split][record_id]
                if official_label != label:
                    raise ValueError(
                        f"Balanced label mismatch at record {record_id}"
                    )
                key = (
                    "test_balanced" if target_split == "test" else target_split
                )
                features[key].append(vector)
                labels[key].append(label)
                metadata[key].append(
                    {
                        "record_id": record_id,
                        "cycle_no": cycle,
                        "case_type": None,
                    }
                )
            if split == "test":
                features["test_natural"].append(vector)
                labels["test_natural"].append(label)
                metadata["test_natural"].append(
                    {
                        "record_id": record_id,
                        "cycle_no": cycle,
                        "case_type": None,
                    }
                )
            previous_by_cycle[cycle] = item

    arrays = {}
    for name in features:
        arrays[name] = (
            np.asarray(features[name], dtype=np.float64),
            np.asarray(labels[name], dtype=bool),
            metadata[name],
        )
    del features, labels

    expected = {
        "train": 24762,
        "val": 5306,
        "test_balanced": 5310,
        "test_natural": 92194,
    }
    for name, count in expected.items():
        if arrays[name][0].shape != (count, len(FEATURE_NAMES)):
            raise ValueError(
                f"Feature shape mismatch for {name}: {arrays[name][0].shape}"
            )

    x_boundary, y_boundary, meta_boundary = load_boundary(boundary_path)
    if x_boundary.shape != (71, len(FEATURE_NAMES)):
        raise ValueError(f"Boundary shape mismatch: {x_boundary.shape}")

    np.savez_compressed(
        output_dir / "feature_data.npz",
        feature_names=np.asarray(FEATURE_NAMES),
        train_x=arrays["train"][0],
        train_y=arrays["train"][1],
        val_x=arrays["val"][0],
        val_y=arrays["val"][1],
        test_balanced_x=arrays["test_balanced"][0],
        test_balanced_y=arrays["test_balanced"][1],
        test_natural_x=arrays["test_natural"][0],
        test_natural_y=arrays["test_natural"][1],
        boundary_x=x_boundary,
        boundary_y=y_boundary,
    )
    extraction_seconds = time.perf_counter() - extraction_started

    dataset_map = {
        "balanced": arrays["test_balanced"],
        "natural": arrays["test_natural"],
        "boundary": (x_boundary, y_boundary, meta_boundary),
    }
    comparison_rows = []
    run_manifest = {
        "version": "revision_traditional_baselines_v1",
        "seed": SEED,
        "feature_names": FEATURE_NAMES,
        "feature_policy": {
            "source": (
                "Only numeric state/current/previous/difference values already "
                "present in the LoRA prompt."
            ),
            "previous_scope": "within_cycle",
            "cycle_start": (
                "Previous values are imputed with current values, differences with "
                "zero, and has_previous_in_cycle is set to zero."
            ),
            "total_cell_redundancy": (
                "Both fields are retained to match the LoRA prompt; known exact "
                "redundancy is reported separately."
            ),
        },
        "sources": {
            "observed_records": str(source),
            "grouped_dataset": str(grouped_dir),
            "split_design": str(design_path),
            "boundary_set": str(boundary_path),
        },
        "counts": {
            "source": source_count,
            **{name: int(value[0].shape[0]) for name, value in arrays.items()},
            "boundary": int(x_boundary.shape[0]),
        },
        "rule_trigger_counts": dict(rule_counts),
        "feature_extraction_seconds": extraction_seconds,
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "models": {},
    }

    # Isolation Forest: train only on normal records from the same balanced training set.
    x_train, y_train, _ = arrays["train"]
    x_val, y_val, _ = arrays["val"]
    scaler = StandardScaler()
    normal_train = x_train[~y_train]
    scaler.fit(normal_train)
    iforest = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination="auto",
        random_state=SEED,
        n_jobs=-1,
    )
    start = time.perf_counter()
    iforest.fit(scaler.transform(normal_train))
    iforest_fit_seconds = time.perf_counter() - start
    val_scores = -iforest.score_samples(scaler.transform(x_val))
    iforest_threshold = tune_threshold(y_val, val_scores)
    joblib.dump(
        {"scaler": scaler, "model": iforest, "threshold": iforest_threshold},
        output_dir / "isolation_forest.joblib",
    )
    model_metrics = {}
    for dataset_name in DATASET_ORDER:
        x_test, y_test, row_metadata = dataset_map[dataset_name]
        start = time.perf_counter()
        scores = -iforest.score_samples(scaler.transform(x_test))
        elapsed = time.perf_counter() - start
        current_metrics = metrics(
            y_test, scores, iforest_threshold["threshold"]
        )
        current_metrics.update(
            {
                "elapsed_seconds": elapsed,
                "samples_per_second": len(y_test) / elapsed,
            }
        )
        model_metrics[dataset_name] = current_metrics
        write_predictions(
            output_dir / f"isolation_forest_{dataset_name}.jsonl",
            row_metadata,
            y_test,
            scores,
            iforest_threshold["threshold"],
        )
        (output_dir / f"isolation_forest_{dataset_name}.metrics.json").write_text(
            json.dumps(current_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        comparison_rows.append(
            {"model": "Isolation Forest", "dataset": dataset_name, **current_metrics}
        )
    run_manifest["models"]["isolation_forest"] = {
        "training_rows": int(normal_train.shape[0]),
        "training_policy": "normal rows only from balanced training IDs",
        "parameters": iforest.get_params(),
        "threshold_selection": iforest_threshold,
        "fit_seconds": iforest_fit_seconds,
        "metrics": model_metrics,
    }

    # Supervised Random Forest: use exactly the balanced train/validation record IDs.
    random_forest = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    start = time.perf_counter()
    random_forest.fit(x_train, y_train)
    rf_fit_seconds = time.perf_counter() - start
    val_scores = random_forest.predict_proba(x_val)[:, 1]
    rf_threshold = tune_threshold(y_val, val_scores)
    joblib.dump(
        {"model": random_forest, "threshold": rf_threshold},
        output_dir / "random_forest.joblib",
    )
    model_metrics = {}
    for dataset_name in DATASET_ORDER:
        x_test, y_test, row_metadata = dataset_map[dataset_name]
        start = time.perf_counter()
        scores = random_forest.predict_proba(x_test)[:, 1]
        elapsed = time.perf_counter() - start
        current_metrics = metrics(y_test, scores, rf_threshold["threshold"])
        current_metrics.update(
            {
                "elapsed_seconds": elapsed,
                "samples_per_second": len(y_test) / elapsed,
            }
        )
        model_metrics[dataset_name] = current_metrics
        write_predictions(
            output_dir / f"random_forest_{dataset_name}.jsonl",
            row_metadata,
            y_test,
            scores,
            rf_threshold["threshold"],
        )
        (output_dir / f"random_forest_{dataset_name}.metrics.json").write_text(
            json.dumps(current_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        comparison_rows.append(
            {"model": "Random Forest", "dataset": dataset_name, **current_metrics}
        )
    importances = sorted(
        zip(FEATURE_NAMES, random_forest.feature_importances_),
        key=lambda pair: pair[1],
        reverse=True,
    )
    write_importance = [
        {"feature": name, "importance": float(value)}
        for name, value in importances
    ]
    with (output_dir / "random_forest_feature_importance.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "importance"])
        writer.writeheader()
        writer.writerows(write_importance)
    run_manifest["models"]["random_forest"] = {
        "training_rows": int(x_train.shape[0]),
        "training_policy": "all balanced training IDs with class_weight=balanced",
        "parameters": random_forest.get_params(),
        "threshold_selection": rf_threshold,
        "fit_seconds": rf_fit_seconds,
        "feature_importance": write_importance,
        "metrics": model_metrics,
    }
    run_manifest["peak_process_memory_mib"] = peak_memory_mib()

    comparison_fields = [
        "model",
        "dataset",
        "samples",
        "prevalence",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
        "fnr",
        "specificity",
        "auroc",
        "auprc",
        "tp",
        "fp",
        "tn",
        "fn",
        "elapsed_seconds",
        "samples_per_second",
    ]
    with (output_dir / "comparison_table.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    provenance = {"version": "revision_traditional_baselines_v1", "files": {}}
    for path in sorted(output_dir.iterdir()):
        if path.name == "provenance_manifest.json" or not path.is_file():
            continue
        provenance["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    (output_dir / "provenance_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_dir.rename(final_output_dir)
    print(json.dumps(run_manifest["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(comparison_rows, ensure_ascii=False, indent=2))
    print(f"OUTPUT_DIR={final_output_dir}")


if __name__ == "__main__":
    main()
