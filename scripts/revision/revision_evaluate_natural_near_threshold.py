from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


RULES = {
    "charge_current_jump": {
        "feature": "current_diff",
        "threshold": 0.5,
        "window_half_width": 0.01,
        "expected_test_count": 43,
        "applicability": "state=110",
    },
    "cell_voltage_jump": {
        "feature": "cell_voltage_diff",
        "threshold": 0.05,
        "window_half_width": 0.001,
        "expected_test_count": 40,
        "applicability": "all_states",
    },
}
METHODS = {
    "Base-LoRA": {
        "family": "LoRA",
        "directory": "outputs/revision_grouped_v1/eval",
        "prefix": "base_lora_seed42",
    },
    "Pure augmentation": {
        "family": "LoRA",
        "directory": "outputs/revision_grouped_v1/eval",
        "prefix": "pure_augmentation_seed42",
    },
    "Boundary weighted": {
        "family": "LoRA",
        "directory": "outputs/revision_grouped_v1/eval",
        "prefix": "boundary_weighted_seed42",
    },
    "Isolation Forest": {
        "family": "traditional",
        "directory": "outputs/revision_traditional_baselines_v1",
        "prefix": "isolation_forest",
    },
    "Random Forest": {
        "family": "traditional",
        "directory": "outputs/revision_traditional_baselines_v1",
        "prefix": "random_forest",
    },
    "LSTM": {
        "family": "deep temporal",
        "directory": "outputs/revision_lstm_baseline_v3",
        "prefix": "lstm",
    },
    "Qwen zero-shot": {
        "family": "general LLM",
        "directory": "outputs/revision_general_llm_baselines_v1",
        "prefix": "zero_shot",
    },
    "Qwen four-shot": {
        "family": "general LLM",
        "directory": "outputs/revision_general_llm_baselines_v1",
        "prefix": "few_shot",
    },
}
LORA_METHODS = ["Base-LoRA", "Pure augmentation", "Boundary weighted"]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(first, second, tolerance: float = 1e-12) -> bool:
    if first is None or second is None:
        return first is second
    return math.isclose(
        float(first), float(second), rel_tol=tolerance, abs_tol=tolerance
    )


def clopper_pearson(
    correct: int, total: int, alpha: float = 0.05
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Clopper-Pearson interval requires total > 0")
    lower = (
        0.0
        if correct == 0
        else float(beta.ppf(alpha / 2, correct, total - correct + 1))
    )
    upper = (
        1.0
        if correct == total
        else float(beta.ppf(1 - alpha / 2, correct + 1, total - correct))
    )
    return lower, upper


def metrics(rows: list[dict], threshold: float) -> dict:
    if not rows:
        raise AssertionError("Cannot compute metrics for an empty subset")
    gold = np.asarray([bool(row["gold"]) for row in rows], dtype=bool)
    scores = np.asarray(
        [float(row["anomaly_score"]) for row in rows], dtype=float
    )
    stored_pred = np.asarray([bool(row["pred"]) for row in rows], dtype=bool)
    score_pred = scores >= threshold
    if not np.array_equal(stored_pred, score_pred):
        raise AssertionError("Stored predictions do not match score threshold")
    if not np.isfinite(scores).all():
        raise AssertionError("Nonfinite anomaly score")
    tn, fp, fn, tp = confusion_matrix(
        gold, stored_pred, labels=[False, True]
    ).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, stored_pred, average="binary", zero_division=0
    )
    correct = int(tp + tn)
    lower, upper = clopper_pearson(correct, len(rows))
    if np.unique(gold).size == 2:
        auroc = float(roc_auc_score(gold, scores))
        auprc = float(average_precision_score(gold, scores))
    else:
        auroc = None
        auprc = None
    return {
        "samples": len(rows),
        "prevalence": float(gold.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(gold, stored_pred)),
        "accuracy_exact_95ci_lower": lower,
        "accuracy_exact_95ci_upper": upper,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "fnr": float(fn / (fn + tp)) if fn + tp else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "auroc": auroc,
        "auprc": auprc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def exact_mcnemar(first: list[dict], second: list[dict]) -> dict:
    first_by_id = {int(row["record_id"]): row for row in first}
    second_by_id = {int(row["record_id"]): row for row in second}
    if set(first_by_id) != set(second_by_id):
        raise AssertionError("McNemar ID mismatch")
    first_only = 0
    second_only = 0
    for record_id in first_by_id:
        a = first_by_id[record_id]
        b = second_by_id[record_id]
        if bool(a["gold"]) != bool(b["gold"]):
            raise AssertionError("McNemar gold mismatch")
        a_correct = bool(a["pred"]) == bool(a["gold"])
        b_correct = bool(b["pred"]) == bool(b["gold"])
        first_only += int(a_correct and not b_correct)
        second_only += int(b_correct and not a_correct)
    discordant = first_only + second_only
    p_value = (
        float(binomtest(first_only, discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "first_correct_second_wrong": first_only,
        "first_wrong_second_correct": second_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def holm_adjust(rows: list[dict]) -> None:
    ordered = sorted(
        enumerate(rows), key=lambda item: item[1]["exact_two_sided_p"]
    )
    running = 0.0
    total = len(rows)
    for rank, (original_index, row) in enumerate(ordered, 1):
        raw_adjusted = min(
            1.0, (total - rank + 1) * row["exact_two_sided_p"]
        )
        running = max(running, raw_adjusted)
        rows[original_index]["holm_rank"] = rank
        rows[original_index]["holm_adjusted_p"] = running


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise AssertionError(f"No rows for {path.name}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: revision_evaluate_natural_near_threshold.py PROJECT_DIR"
        )
    project = Path(sys.argv[1]).resolve()
    final_dir = project / "outputs/revision_natural_near_threshold_v1"
    output_dir = project / "outputs/revision_natural_near_threshold_v1.tmp"
    if final_dir.exists() or output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {final_dir} or {output_dir}")
    output_dir.mkdir(parents=True)

    reference_path = (
        project / "data/processed/revision_grouped_v1/test_natural.jsonl"
    )
    rule_prediction_path = (
        project
        / "outputs/revision_rule_verifier_grouped_v1/"
        "rule_predictions_natural.jsonl"
    )
    references = read_jsonl(reference_path)
    rule_predictions = read_jsonl(rule_prediction_path)
    reference_by_id = {
        int(row["record_id"]): row for row in references
    }
    rule_by_id = {
        int(row["record_id"]): row for row in rule_predictions
    }
    if (
        len(reference_by_id) != len(references)
        or len(rule_by_id) != len(rule_predictions)
        or set(reference_by_id) != set(rule_by_id)
    ):
        raise AssertionError("Reference/RuleVerifier ID alignment failure")

    selected_records = []
    selected_ids: dict[tuple[str, str], set[int]] = {}
    subset_manifest = []
    overlap_by_rule = {}
    for rule_name, spec in RULES.items():
        selected = []
        for record_id, parsed in rule_by_id.items():
            feature_value = parsed.get(spec["feature"])
            state = parsed.get("state")
            applicable = (
                state == 110
                if spec["applicability"] == "state=110"
                else True
            )
            if (
                not applicable
                or feature_value is None
                or abs(float(feature_value) - spec["threshold"])
                > spec["window_half_width"] + 1e-12
            ):
                continue
            reference = reference_by_id[record_id]
            triggers = list(reference.get("rule_triggers", []))
            target_trigger = float(feature_value) > spec["threshold"]
            if (rule_name in triggers) != target_trigger:
                raise AssertionError(
                    f"Target trigger mismatch: {rule_name} {record_id}"
                )
            gold = bool(reference["is_anomaly"])
            if gold != bool(triggers):
                raise AssertionError(f"Gold/trigger mismatch: {record_id}")
            other_triggers = [
                trigger for trigger in triggers if trigger != rule_name
            ]
            side = (
                "below_or_equal"
                if float(feature_value) <= spec["threshold"]
                else "above"
            )
            selected.append(
                {
                    "rule": rule_name,
                    "record_id": record_id,
                    "cycle_no": str(reference["cycle_no"]),
                    "feature": spec["feature"],
                    "feature_value": float(feature_value),
                    "threshold": spec["threshold"],
                    "window_half_width": spec["window_half_width"],
                    "distance_to_threshold": abs(
                        float(feature_value) - spec["threshold"]
                    ),
                    "side": side,
                    "target_rule_triggered": target_trigger,
                    "other_rule_triggers": other_triggers,
                    "other_rule_trigger_count": len(other_triggers),
                    "isolated_target_rule": len(other_triggers) == 0,
                    "gold": gold,
                    "state": state,
                }
            )
        if len(selected) != spec["expected_test_count"]:
            raise AssertionError(
                f"Near-threshold count mismatch for {rule_name}: "
                f"{len(selected)} != {spec['expected_test_count']}"
            )
        selected_records.extend(selected)
        all_ids = {row["record_id"] for row in selected}
        isolated_ids = {
            row["record_id"]
            for row in selected
            if row["isolated_target_rule"]
        }
        selected_ids[(rule_name, "all")] = all_ids
        selected_ids[(rule_name, "isolated")] = isolated_ids
        overlap_by_rule[rule_name] = len(all_ids)
        for scope, ids in [("all", all_ids), ("isolated", isolated_ids)]:
            current = [row for row in selected if row["record_id"] in ids]
            subset_manifest.append(
                {
                    "rule": rule_name,
                    "scope": scope,
                    "samples": len(current),
                    "below_or_equal": sum(
                        row["side"] == "below_or_equal" for row in current
                    ),
                    "above": sum(row["side"] == "above" for row in current),
                    "gold_normal": sum(not row["gold"] for row in current),
                    "gold_anomaly": sum(row["gold"] for row in current),
                    "records_with_other_rule_triggers": sum(
                        row["other_rule_trigger_count"] > 0 for row in current
                    ),
                }
            )

    intersection = (
        selected_ids[("charge_current_jump", "all")]
        & selected_ids[("cell_voltage_jump", "all")]
    )

    prediction_cache: dict[str, dict[int, dict]] = {}
    threshold_by_method = {}
    input_manifest = {
        "reference": {
            "path": str(reference_path),
            "bytes": reference_path.stat().st_size,
            "sha256": sha256(reference_path),
            "rows": len(references),
        },
        "rule_predictions": {
            "path": str(rule_prediction_path),
            "bytes": rule_prediction_path.stat().st_size,
            "sha256": sha256(rule_prediction_path),
            "rows": len(rule_predictions),
        },
    }
    for method, config in METHODS.items():
        directory = project / config["directory"]
        prediction_path = directory / f"{config['prefix']}_natural.jsonl"
        metric_path = directory / f"{config['prefix']}_natural.metrics.json"
        rows = read_jsonl(prediction_path)
        stored_metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        by_id = {int(row["record_id"]): row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != set(reference_by_id):
            raise AssertionError(f"Prediction ID mismatch: {method}")
        for record_id, row in by_id.items():
            if bool(row["gold"]) != bool(
                reference_by_id[record_id]["is_anomaly"]
            ):
                raise AssertionError(f"Prediction gold mismatch: {method}")
        prediction_cache[method] = by_id
        threshold_by_method[method] = float(stored_metrics["threshold"])
        input_manifest[f"prediction_{method}"] = {
            "path": str(prediction_path),
            "bytes": prediction_path.stat().st_size,
            "sha256": sha256(prediction_path),
            "rows": len(rows),
            "threshold": threshold_by_method[method],
        }

    metric_rows = []
    side_rows = []
    subset_predictions: dict[tuple[str, str, str], list[dict]] = {}
    for method, config in METHODS.items():
        for rule_name in RULES:
            for scope in ["all", "isolated"]:
                ids = selected_ids[(rule_name, scope)]
                rows = [prediction_cache[method][record_id] for record_id in ids]
                calculated = metrics(rows, threshold_by_method[method])
                metric_rows.append(
                    {
                        "method": method,
                        "family": config["family"],
                        "rule": rule_name,
                        "scope": scope,
                        **calculated,
                    }
                )
                subset_predictions[(method, rule_name, scope)] = rows
                record_meta = {
                    row["record_id"]: row
                    for row in selected_records
                    if row["rule"] == rule_name
                }
                for side in ["below_or_equal", "above"]:
                    side_prediction_rows = [
                        row
                        for row in rows
                        if record_meta[int(row["record_id"])]["side"] == side
                    ]
                    if not side_prediction_rows:
                        side_rows.append(
                            {
                                "method": method,
                                "family": config["family"],
                                "rule": rule_name,
                                "scope": scope,
                                "side": side,
                                "samples": 0,
                                "correct": 0,
                                "accuracy": None,
                                "accuracy_exact_95ci_lower": None,
                                "accuracy_exact_95ci_upper": None,
                            }
                        )
                        continue
                    correct = sum(
                        bool(row["pred"]) == bool(row["gold"])
                        for row in side_prediction_rows
                    )
                    lower, upper = clopper_pearson(
                        correct, len(side_prediction_rows)
                    )
                    side_rows.append(
                        {
                            "method": method,
                            "family": config["family"],
                            "rule": rule_name,
                            "scope": scope,
                            "side": side,
                            "samples": len(side_prediction_rows),
                            "correct": correct,
                            "accuracy": correct / len(side_prediction_rows),
                            "accuracy_exact_95ci_lower": lower,
                            "accuracy_exact_95ci_upper": upper,
                        }
                    )

    pairwise_rows = []
    for rule_name in RULES:
        for scope in ["all", "isolated"]:
            family_rows = []
            for first, second in combinations(LORA_METHODS, 2):
                result = exact_mcnemar(
                    subset_predictions[(first, rule_name, scope)],
                    subset_predictions[(second, rule_name, scope)],
                )
                family_rows.append(
                    {
                        "rule": rule_name,
                        "scope": scope,
                        "first": first,
                        "second": second,
                        **result,
                    }
                )
            holm_adjust(family_rows)
            pairwise_rows.extend(family_rows)

    rv_inconsistent_path = (
        project
        / "outputs/revision_rule_verifier_grouped_v1/"
        "inconsistent_cases.jsonl"
    )
    rv_inconsistent = [
        row
        for row in read_jsonl(rv_inconsistent_path)
        if row["dataset"] == "natural"
    ]
    rv_keys = {
        (row["method"], int(row["record_id"])) for row in rv_inconsistent
    }
    rv_rows = []
    for method in LORA_METHODS:
        for rule_name in RULES:
            for scope in ["all", "isolated"]:
                ids = selected_ids[(rule_name, scope)]
                predictions = subset_predictions[(method, rule_name, scope)]
                model_errors = sum(
                    bool(row["pred"]) != bool(row["gold"])
                    for row in predictions
                )
                corrections = sum(
                    (method, record_id) in rv_keys for record_id in ids
                )
                if model_errors != corrections:
                    raise AssertionError(
                        f"RuleVerifier correction mismatch: "
                        f"{method} {rule_name} {scope}"
                    )
                rv_rows.append(
                    {
                        "method": method,
                        "rule": rule_name,
                        "scope": scope,
                        "samples": len(ids),
                        "model_errors": model_errors,
                        "rule_verifier_corrections": corrections,
                        "unnecessary_overrides": 0,
                        "final_errors": 0,
                    }
                )

    selected_records.sort(key=lambda row: (row["rule"], row["record_id"]))
    with (output_dir / "selected_records.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in selected_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(output_dir / "subset_manifest.csv", subset_manifest)
    write_csv(output_dir / "cross_method_metrics.csv", metric_rows)
    write_csv(output_dir / "side_accuracy.csv", side_rows)
    write_csv(output_dir / "lora_pairwise_mcnemar.csv", pairwise_rows)
    write_csv(output_dir / "rule_verifier_corrections.csv", rv_rows)

    by_key = {
        (row["method"], row["rule"], row["scope"]): row
        for row in metric_rows
    }
    lines = [
        "# Natural near-threshold evaluation",
        "",
        "## Selection audit",
        "",
        f"- Charge-current primary band: {len(selected_ids[('charge_current_jump', 'all')])} "
        "held-out natural-test records at 0.5 ± 0.01 A.",
        f"- Cell-voltage primary band: {len(selected_ids[('cell_voltage_jump', 'all')])} "
        "held-out natural-test records at 0.05 ± 0.001 V.",
        f"- Overlap between the two bands: {len(intersection)} unique records.",
        "- `all` includes records that may trigger other rules; `isolated` removes "
        "records with any non-target rule trigger.",
        "",
        "## All near-threshold records",
        "",
        "| Method | Current N | Current accuracy | Current FP/FN | "
        "Cell N | Cell accuracy | Cell FP/FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        current = by_key[(method, "charge_current_jump", "all")]
        cell = by_key[(method, "cell_voltage_jump", "all")]
        lines.append(
            f"| {method} | {current['samples']} | {current['accuracy']:.4%} | "
            f"{current['fp']}/{current['fn']} | {cell['samples']} | "
            f"{cell['accuracy']:.4%} | {cell['fp']}/{cell['fn']} |"
        )
    lines.extend(
        [
            "",
            "## Isolated target-rule records",
            "",
            "| Method | Current N | Current accuracy | Current FP/FN | "
            "Cell N | Cell accuracy | Cell FP/FN |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        current = by_key[(method, "charge_current_jump", "isolated")]
        cell = by_key[(method, "cell_voltage_jump", "isolated")]
        lines.append(
            f"| {method} | {current['samples']} | {current['accuracy']:.4%} | "
            f"{current['fp']}/{current['fn']} | {cell['samples']} | "
            f"{cell['accuracy']:.4%} | {cell['fp']}/{cell['fn']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "These subsets provide limited in-source evidence on naturally observed "
            "near-threshold records from held-out cycles. They do not establish "
            "cross-device, cross-domain, or industrial field generalization. "
            "Temperature and both total-voltage rule boundaries remain covered "
            "only by synthetic BoundarySet cases.",
            "",
        ]
    )
    (output_dir / "comparison_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    manifest = {
        "version": "revision_natural_near_threshold_v1",
        "selection": RULES,
        "checks": {
            "expected_43_current_records": "pass",
            "expected_40_cell_records": "pass",
            "reference_rule_prediction_id_alignment": "pass",
            "prediction_id_alignment_all_8_methods": "pass",
            "prediction_gold_alignment_all_8_methods": "pass",
            "stored_pred_matches_continuous_score_threshold": "pass",
            "target_trigger_matches_strict_threshold": "pass",
            "gold_matches_any_rule_trigger": "pass",
            "rule_verifier_corrections_match_model_errors": "pass",
        },
        "overlap_between_rule_bands": len(intersection),
        "subset_manifest": subset_manifest,
        "methods": METHODS,
        "inputs": input_manifest,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    provenance = {
        "version": "revision_natural_near_threshold_v1",
        "files": {},
    }
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
    output_dir.rename(final_dir)
    print(json.dumps(manifest["checks"], ensure_ascii=False, indent=2))
    print("\n".join(lines))
    print(f"OUTPUT_DIR={final_dir}")


if __name__ == "__main__":
    main()
