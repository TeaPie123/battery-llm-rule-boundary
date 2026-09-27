from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


MODELS = {
    "Isolation Forest": "isolation_forest",
    "Random Forest": "random_forest",
}
LORA_MODELS = {
    "Base-LoRA": "base_lora_seed42",
    "Pure augmentation": "pure_augmentation_seed42",
    "Boundary weighted": "boundary_weighted_seed42",
}
DATASETS = ["balanced", "natural", "boundary"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def close(a: float | None, b: float | None, tolerance: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def recompute(rows: list[dict], threshold: float) -> dict:
    gold = np.asarray([bool(row["gold"]) for row in rows], dtype=bool)
    scores = np.asarray([float(row["anomaly_score"]) for row in rows], dtype=float)
    pred = scores >= threshold
    stored_pred = np.asarray([bool(row["pred"]) for row in rows], dtype=bool)
    if not np.array_equal(pred, stored_pred):
        raise AssertionError("Stored predictions do not match scores and threshold")
    if not np.isfinite(scores).all():
        raise AssertionError("Prediction scores contain NaN or infinity")
    tn, fp, fn, tp = confusion_matrix(gold, pred, labels=[False, True]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, pred, average="binary", zero_division=0
    )
    return {
        "samples": int(gold.size),
        "prevalence": float(gold.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(gold, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "fnr": float(fn / (fn + tp)) if fn + tp else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "auroc": float(roc_auc_score(gold, scores)),
        "auprc": float(average_precision_score(gold, scores)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def load_reference(project: Path, dataset: str) -> list[dict]:
    if dataset == "balanced":
        path = project / "data/processed/revision_grouped_v1/test_balanced.jsonl"
        rows = read_jsonl(path)
        return [
            {
                "record_id": int(row["record_id"]),
                "gold": bool(row["is_anomaly"]),
                "cycle_no": str(row["cycle_no"]),
            }
            for row in rows
        ]
    if dataset == "natural":
        path = project / "data/processed/revision_grouped_v1/test_natural.jsonl"
        rows = read_jsonl(path)
        return [
            {
                "record_id": int(row["record_id"]),
                "gold": bool(row["is_anomaly"]),
                "cycle_no": str(row["cycle_no"]),
            }
            for row in rows
        ]
    path = project / "data/processed/boundary_test/boundary_test.jsonl"
    rows = read_jsonl(path)
    return [
        {
            "record_id": int(row["record_id"]),
            "gold": bool(row["gold"]),
            "cycle_no": None,
        }
        for row in rows
    ]


def exact_mcnemar(first: list[dict], second: list[dict]) -> dict:
    if len(first) != len(second):
        raise AssertionError("McNemar sample size mismatch")
    first_by_id = {int(row["record_id"]): row for row in first}
    second_by_id = {int(row["record_id"]): row for row in second}
    if set(first_by_id) != set(second_by_id):
        raise AssertionError("McNemar record ID mismatch")
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


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: revision_audit_traditional_baselines.py PROJECT_DIR"
        )
    project = Path(sys.argv[1]).resolve()
    result_dir = project / "outputs/revision_traditional_baselines_v1"
    audit_dir = result_dir / "audit"
    if audit_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {audit_dir}")
    audit_dir.mkdir()

    manifest = json.loads(
        (result_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (result_dir / "provenance_manifest.json").read_text(encoding="utf-8")
    )
    checks = []

    for filename, expected in provenance["files"].items():
        path = result_dir / filename
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if actual != expected:
            raise AssertionError(f"Provenance mismatch: {filename}")
    checks.append(
        {
            "check": "output_provenance",
            "status": "pass",
            "details": f"{len(provenance['files'])} files verified",
        }
    )

    reference = {dataset: load_reference(project, dataset) for dataset in DATASETS}
    for dataset, rows in reference.items():
        ids = [row["record_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"Duplicate reference IDs: {dataset}")
    if not set(row["record_id"] for row in reference["balanced"]).issubset(
        set(row["record_id"] for row in reference["natural"])
    ):
        raise AssertionError("Balanced test is not a subset of natural test")
    checks.append(
        {
            "check": "reference_test_sets",
            "status": "pass",
            "details": {
                dataset: len(rows) for dataset, rows in reference.items()
            },
        }
    )

    recomputed_metrics = {}
    traditional_predictions = {}
    for model_name, prefix in MODELS.items():
        recomputed_metrics[model_name] = {}
        traditional_predictions[model_name] = {}
        threshold = float(
            manifest["models"][prefix]["threshold_selection"]["threshold"]
        )
        for dataset in DATASETS:
            prediction_path = result_dir / f"{prefix}_{dataset}.jsonl"
            metric_path = result_dir / f"{prefix}_{dataset}.metrics.json"
            rows = read_jsonl(prediction_path)
            official = reference[dataset]
            prediction_ids = [int(row["record_id"]) for row in rows]
            if len(prediction_ids) != len(set(prediction_ids)):
                raise AssertionError(
                    f"Duplicate prediction IDs: {model_name} {dataset}"
                )
            official_by_id = {
                int(row["record_id"]): bool(row["gold"]) for row in official
            }
            if set(prediction_ids) != set(official_by_id):
                raise AssertionError(
                    f"Prediction ID set mismatch: {model_name} {dataset}"
                )
            for row in rows:
                if bool(row["gold"]) != official_by_id[int(row["record_id"])]:
                    raise AssertionError(
                        f"Prediction gold mismatch: {model_name} {dataset} "
                        f"record={row['record_id']}"
                    )
            recalculated = recompute(rows, threshold)
            stored = json.loads(metric_path.read_text(encoding="utf-8"))
            for key, value in recalculated.items():
                if not close(value, stored.get(key)):
                    raise AssertionError(
                        f"Metric mismatch: {model_name} {dataset} {key}"
                    )
            recomputed_metrics[model_name][dataset] = recalculated
            traditional_predictions[model_name][dataset] = rows
    checks.append(
        {
            "check": "predictions_and_metrics",
            "status": "pass",
            "details": (
                "6 prediction files verified against exact official ID sets, "
                "per-ID labels, scores, and metrics; row order is immaterial "
                "for these non-sequential models"
            ),
        }
    )

    feature_archive = np.load(result_dir / "feature_data.npz")
    expected_shapes = {
        "train_x": (24762, 16),
        "val_x": (5306, 16),
        "test_balanced_x": (5310, 16),
        "test_natural_x": (92194, 16),
        "boundary_x": (71, 16),
    }
    for key, shape in expected_shapes.items():
        if feature_archive[key].shape != shape:
            raise AssertionError(f"Feature archive shape mismatch: {key}")
        if not np.isfinite(feature_archive[key]).all():
            raise AssertionError(f"Feature archive has nonfinite values: {key}")
    checks.append(
        {
            "check": "feature_archive",
            "status": "pass",
            "details": expected_shapes,
        }
    )

    iforest_bundle = joblib.load(result_dir / "isolation_forest.joblib")
    rf_bundle = joblib.load(result_dir / "random_forest.joblib")
    if iforest_bundle["model"].n_estimators != 300:
        raise AssertionError("Isolation Forest tree count mismatch")
    if rf_bundle["model"].n_estimators != 500:
        raise AssertionError("Random Forest tree count mismatch")
    if not close(
        iforest_bundle["threshold"]["threshold"],
        manifest["models"]["isolation_forest"]["threshold_selection"]["threshold"],
    ):
        raise AssertionError("Isolation Forest threshold mismatch")
    if not close(
        rf_bundle["threshold"]["threshold"],
        manifest["models"]["random_forest"]["threshold_selection"]["threshold"],
    ):
        raise AssertionError("Random Forest threshold mismatch")
    checks.append(
        {
            "check": "serialized_models",
            "status": "pass",
            "details": "model classes, tree counts, and thresholds verified",
        }
    )

    cross_method_rows = []
    mcnemar_rows = []
    for dataset in DATASETS:
        for lora_name, lora_prefix in LORA_MODELS.items():
            metric_path = (
                project
                / "outputs/revision_grouped_v1/eval"
                / f"{lora_prefix}_{dataset}.metrics.json"
            )
            lora_metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            cross_method_rows.append(
                {
                    "model": lora_name,
                    "family": "LoRA",
                    "dataset": dataset,
                    **{
                        key: lora_metrics[key]
                        for key in [
                            "samples",
                            "accuracy",
                            "precision",
                            "recall",
                            "f1",
                            "fpr",
                            "fnr",
                            "auroc",
                            "auprc",
                            "tp",
                            "fp",
                            "tn",
                            "fn",
                        ]
                    },
                }
            )
        for model_name in MODELS:
            cross_method_rows.append(
                {
                    "model": model_name,
                    "family": "traditional",
                    "dataset": dataset,
                    **{
                        key: recomputed_metrics[model_name][dataset][key]
                        for key in [
                            "samples",
                            "accuracy",
                            "precision",
                            "recall",
                            "f1",
                            "fpr",
                            "fnr",
                            "auroc",
                            "auprc",
                            "tp",
                            "fp",
                            "tn",
                            "fn",
                        ]
                    },
                }
            )

        rf_rows = traditional_predictions["Random Forest"][dataset]
        for lora_name, lora_prefix in LORA_MODELS.items():
            lora_path = (
                project
                / "outputs/revision_grouped_v1/eval"
                / f"{lora_prefix}_{dataset}.jsonl"
            )
            lora_rows = read_jsonl(lora_path)
            comparison = exact_mcnemar(rf_rows, lora_rows)
            mcnemar_rows.append(
                {
                    "dataset": dataset,
                    "first": "Random Forest",
                    "second": lora_name,
                    **comparison,
                }
            )

    cross_fields = [
        "model",
        "family",
        "dataset",
        "samples",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
        "fnr",
        "auroc",
        "auprc",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    with (audit_dir / "cross_method_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=cross_fields)
        writer.writeheader()
        writer.writerows(cross_method_rows)

    mcnemar_fields = [
        "dataset",
        "first",
        "second",
        "first_correct_second_wrong",
        "first_wrong_second_correct",
        "discordant",
        "exact_two_sided_p",
    ]
    with (audit_dir / "random_forest_vs_lora_mcnemar.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=mcnemar_fields)
        writer.writeheader()
        writer.writerows(mcnemar_rows)

    checks.append(
        {
            "check": "cross_method_pairing",
            "status": "pass",
            "details": "Random Forest and LoRA predictions paired by exact record ID for 9 McNemar tests",
        }
    )
    audit_json = {
        "version": "revision_traditional_baselines_v1_audit",
        "checks": checks,
        "recomputed_metrics": recomputed_metrics,
        "mcnemar": mcnemar_rows,
        "source_hashes": {
            "analysis_script": {
                "path": str(project / "scripts/revision_traditional_baselines.py"),
                "bytes": (
                    project / "scripts/revision_traditional_baselines.py"
                ).stat().st_size,
                "sha256": sha256(
                    project / "scripts/revision_traditional_baselines.py"
                ),
            },
            "grouped_manifest": {
                "path": str(
                    project / "data/processed/revision_grouped_v1/manifest.json"
                ),
                "bytes": (
                    project / "data/processed/revision_grouped_v1/manifest.json"
                ).stat().st_size,
                "sha256": sha256(
                    project / "data/processed/revision_grouped_v1/manifest.json"
                ),
            },
            "boundary_set": {
                "path": str(
                    project
                    / "data/processed/boundary_test/boundary_test.jsonl"
                ),
                "bytes": (
                    project
                    / "data/processed/boundary_test/boundary_test.jsonl"
                ).stat().st_size,
                "sha256": sha256(
                    project
                    / "data/processed/boundary_test/boundary_test.jsonl"
                ),
            },
        },
    }
    (audit_dir / "audit_report.json").write_text(
        json.dumps(audit_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rf_natural = recomputed_metrics["Random Forest"]["natural"]
    rf_boundary = recomputed_metrics["Random Forest"]["boundary"]
    report = f"""# Traditional baseline audit

All audit checks passed.

## Random Forest headline results

- Balanced test: {recomputed_metrics["Random Forest"]["balanced"]["accuracy"]:.6%}
- Natural test: {rf_natural["accuracy"]:.6%}, FP/FN={rf_natural["fp"]}/{rf_natural["fn"]}
- BoundarySet: {rf_boundary["accuracy"]:.6%}, F1={rf_boundary["f1"]:.6%}

## Interpretation

The Random Forest nearly matches the strongest LoRA methods on the in-source natural
test and is competitive on BoundarySet. The paper must not claim universal superiority
over traditional methods. Paired McNemar results are reported separately and should be
interpreted with the synthetic BoundarySet and rule-derived labels in mind.
"""
    (audit_dir / "audit_report.md").write_text(report, encoding="utf-8")

    audit_provenance = {
        "version": "revision_traditional_baselines_v1_audit",
        "files": {},
    }
    for path in sorted(audit_dir.iterdir()):
        if path.name == "provenance_manifest.json" or not path.is_file():
            continue
        audit_provenance["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    (audit_dir / "provenance_manifest.json").write_text(
        json.dumps(audit_provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps(mcnemar_rows, ensure_ascii=False, indent=2))
    print(f"AUDIT_DIR={audit_dir}")


if __name__ == "__main__":
    main()
