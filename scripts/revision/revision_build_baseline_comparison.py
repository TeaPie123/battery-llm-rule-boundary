from __future__ import annotations

import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta

from revision_audit_traditional_baselines import (
    close,
    exact_mcnemar,
    load_reference,
    read_jsonl,
    recompute,
    sha256,
)


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
}
DATASETS = ["balanced", "natural", "boundary"]


def clopper_pearson(correct: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if correct == 0 else float(beta.ppf(alpha / 2, correct, total - correct + 1))
    upper = 1.0 if correct == total else float(beta.ppf(1 - alpha / 2, correct + 1, total - correct))
    return lower, upper


def verify_provenance(directory: Path) -> int:
    manifest = json.loads(
        (directory / "provenance_manifest.json").read_text(encoding="utf-8")
    )
    for filename, expected in manifest["files"].items():
        path = directory / filename
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if actual != expected:
            raise AssertionError(f"Provenance mismatch: {path}")
    return len(manifest["files"])


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: revision_build_baseline_comparison.py PROJECT_DIR"
        )
    project = Path(sys.argv[1]).resolve()
    final_dir = project / "outputs/revision_baseline_comparison_v1"
    output_dir = project / "outputs/revision_baseline_comparison_v1.tmp"
    if final_dir.exists() or output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {final_dir} or {output_dir}")
    output_dir.mkdir(parents=True)

    traditional_dir = project / "outputs/revision_traditional_baselines_v1"
    lstm_dir = project / "outputs/revision_lstm_baseline_v3"
    provenance_counts = {
        "traditional": verify_provenance(traditional_dir),
        "lstm_v3": verify_provenance(lstm_dir),
    }

    lstm_manifest = json.loads(
        (lstm_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    training_log = read_jsonl(lstm_dir / "training_log.jsonl")
    best_epoch = int(lstm_manifest["model"]["best_epoch"])
    patience = 10
    if training_log[-1]["epoch"] - best_epoch != patience:
        raise AssertionError("LSTM did not terminate by the documented patience")
    if training_log[-1]["epoch"] >= lstm_manifest["model"]["max_epochs"]:
        raise AssertionError("LSTM stopped at max epoch instead of natural early stopping")
    checkpoint = torch.load(
        lstm_dir / "lstm_best.pt", map_location="cpu", weights_only=False
    )
    if int(checkpoint["best_epoch"]) != best_epoch:
        raise AssertionError("LSTM checkpoint best epoch mismatch")
    if not close(
        checkpoint["threshold_selection"]["threshold"],
        lstm_manifest["model"]["threshold_selection"]["threshold"],
    ):
        raise AssertionError("LSTM threshold mismatch")
    sequence_archive = np.load(lstm_dir / "sequence_data.npz")
    expected_shapes = {
        "train_x": (24762, 8, 7),
        "val_x": (5306, 8, 7),
        "test_balanced_x": (5310, 8, 7),
        "test_natural_x": (92194, 8, 7),
        "boundary_x": (71, 8, 7),
    }
    for key, shape in expected_shapes.items():
        if sequence_archive[key].shape != shape:
            raise AssertionError(f"LSTM sequence shape mismatch: {key}")
        if not np.isfinite(sequence_archive[key]).all():
            raise AssertionError(f"LSTM sequence contains nonfinite values: {key}")

    references = {dataset: load_reference(project, dataset) for dataset in DATASETS}
    predictions = {method: {} for method in METHODS}
    metric_rows = []
    ci_rows = []
    audit_checks = []

    for method, config in METHODS.items():
        directory = project / config["directory"]
        prefix = config["prefix"]
        for dataset in DATASETS:
            pred_path = directory / f"{prefix}_{dataset}.jsonl"
            metric_path = directory / f"{prefix}_{dataset}.metrics.json"
            rows = read_jsonl(pred_path)
            stored = json.loads(metric_path.read_text(encoding="utf-8"))
            official = {
                int(row["record_id"]): bool(row["gold"])
                for row in references[dataset]
            }
            current_ids = [int(row["record_id"]) for row in rows]
            if len(current_ids) != len(set(current_ids)):
                raise AssertionError(f"Duplicate prediction IDs: {method} {dataset}")
            if set(current_ids) != set(official):
                raise AssertionError(f"Prediction ID set mismatch: {method} {dataset}")
            for row in rows:
                if bool(row["gold"]) != official[int(row["record_id"])]:
                    raise AssertionError(f"Gold mismatch: {method} {dataset}")
            recalculated = recompute(rows, float(stored["threshold"]))
            for key, value in recalculated.items():
                if not close(value, stored.get(key)):
                    raise AssertionError(
                        f"Stored metric mismatch: {method} {dataset} {key}"
                    )
            predictions[method][dataset] = rows
            metric_rows.append(
                {
                    "method": method,
                    "family": config["family"],
                    "dataset": dataset,
                    **recalculated,
                }
            )
            if dataset == "boundary":
                correct = recalculated["tp"] + recalculated["tn"]
                lower, upper = clopper_pearson(correct, recalculated["samples"])
                ci_rows.append(
                    {
                        "method": method,
                        "correct": correct,
                        "total": recalculated["samples"],
                        "accuracy": recalculated["accuracy"],
                        "exact_95ci_lower": lower,
                        "exact_95ci_upper": upper,
                    }
                )
    audit_checks.append(
        {
            "check": "all_predictions_and_metrics",
            "status": "pass",
            "details": "18 prediction/metric pairs matched official ID sets, per-ID gold labels, scores, thresholds, and recomputed metrics",
        }
    )
    audit_checks.append(
        {
            "check": "lstm_convergence",
            "status": "pass",
            "details": {
                "best_epoch": best_epoch,
                "stopped_epoch": training_log[-1]["epoch"],
                "patience": patience,
                "stopped_before_max_epoch": True,
            },
        }
    )
    audit_checks.append(
        {
            "check": "source_provenance",
            "status": "pass",
            "details": provenance_counts,
        }
    )

    pairwise_rows = []
    for dataset in DATASETS:
        for first, second in combinations(METHODS, 2):
            result = exact_mcnemar(
                predictions[first][dataset], predictions[second][dataset]
            )
            pairwise_rows.append(
                {
                    "dataset": dataset,
                    "first": first,
                    "second": second,
                    **result,
                }
            )
    audit_checks.append(
        {
            "check": "pairwise_mcnemar",
            "status": "pass",
            "details": f"{len(pairwise_rows)} exact paired tests computed by record ID",
        }
    )

    metric_fields = [
        "method",
        "family",
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
    ]
    with (output_dir / "cross_method_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    pairwise_fields = [
        "dataset",
        "first",
        "second",
        "first_correct_second_wrong",
        "first_wrong_second_correct",
        "discordant",
        "exact_two_sided_p",
    ]
    with (output_dir / "pairwise_mcnemar.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=pairwise_fields)
        writer.writeheader()
        writer.writerows(pairwise_rows)

    with (output_dir / "boundary_accuracy_exact_ci.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ci_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ci_rows)

    by_key = {
        (row["method"], row["dataset"]): row for row in metric_rows
    }
    report = f"""# Traditional and temporal baseline comparison

## Audit status

All checks passed. All methods use the same cycle-isolated test record sets and gold labels.
The final LSTM v3 reached its best validation loss at epoch {best_epoch} and stopped naturally
at epoch {training_log[-1]["epoch"]} after patience {patience}; v1/v2 are convergence pilots
and must not be used in the paper.

## Headline results

| Method | Natural accuracy | Natural FP/FN | Natural AUPRC | Boundary accuracy | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| Base-LoRA | {by_key[("Base-LoRA", "natural")]["accuracy"]:.4%} | {by_key[("Base-LoRA", "natural")]["fp"]}/{by_key[("Base-LoRA", "natural")]["fn"]} | {by_key[("Base-LoRA", "natural")]["auprc"]:.4%} | {by_key[("Base-LoRA", "boundary")]["accuracy"]:.4%} | {by_key[("Base-LoRA", "boundary")]["f1"]:.4%} |
| Pure augmentation | {by_key[("Pure augmentation", "natural")]["accuracy"]:.4%} | {by_key[("Pure augmentation", "natural")]["fp"]}/{by_key[("Pure augmentation", "natural")]["fn"]} | {by_key[("Pure augmentation", "natural")]["auprc"]:.4%} | {by_key[("Pure augmentation", "boundary")]["accuracy"]:.4%} | {by_key[("Pure augmentation", "boundary")]["f1"]:.4%} |
| Boundary weighted | {by_key[("Boundary weighted", "natural")]["accuracy"]:.4%} | {by_key[("Boundary weighted", "natural")]["fp"]}/{by_key[("Boundary weighted", "natural")]["fn"]} | {by_key[("Boundary weighted", "natural")]["auprc"]:.4%} | {by_key[("Boundary weighted", "boundary")]["accuracy"]:.4%} | {by_key[("Boundary weighted", "boundary")]["f1"]:.4%} |
| Isolation Forest | {by_key[("Isolation Forest", "natural")]["accuracy"]:.4%} | {by_key[("Isolation Forest", "natural")]["fp"]}/{by_key[("Isolation Forest", "natural")]["fn"]} | {by_key[("Isolation Forest", "natural")]["auprc"]:.4%} | {by_key[("Isolation Forest", "boundary")]["accuracy"]:.4%} | {by_key[("Isolation Forest", "boundary")]["f1"]:.4%} |
| Random Forest | {by_key[("Random Forest", "natural")]["accuracy"]:.4%} | {by_key[("Random Forest", "natural")]["fp"]}/{by_key[("Random Forest", "natural")]["fn"]} | {by_key[("Random Forest", "natural")]["auprc"]:.4%} | {by_key[("Random Forest", "boundary")]["accuracy"]:.4%} | {by_key[("Random Forest", "boundary")]["f1"]:.4%} |
| LSTM | {by_key[("LSTM", "natural")]["accuracy"]:.4%} | {by_key[("LSTM", "natural")]["fp"]}/{by_key[("LSTM", "natural")]["fn"]} | {by_key[("LSTM", "natural")]["auprc"]:.4%} | {by_key[("LSTM", "boundary")]["accuracy"]:.4%} | {by_key[("LSTM", "boundary")]["f1"]:.4%} |

## Interpretation

1. Random Forest is a strong in-source baseline because labels are deterministic threshold rules
   over the same numeric inputs. The paper cannot claim that LoRA universally outperforms
   traditional supervised models.
2. Pure augmentation and boundary weighting reduce natural-test false positives from Random
   Forest's 11 to 1, while keeping zero false negatives.
3. Pure augmentation has the highest BoundarySet point estimate, but Random Forest is competitive;
   all BoundarySet claims remain limited by 71 synthetic cases.
4. LSTM uses longer within-cycle context but performs worse than Random Forest and LoRA, especially
   on BoundarySet. Sequence context alone does not provide strict threshold compliance.
5. Isolation Forest has high recall but excessive natural false positives and poor BoundarySet
   specificity.
"""
    (output_dir / "comparison_report.md").write_text(report, encoding="utf-8")

    audit = {
        "version": "revision_baseline_comparison_v1",
        "checks": audit_checks,
        "methods": METHODS,
        "datasets": {key: len(value) for key, value in references.items()},
        "lstm_final_version": {
            "used": "revision_lstm_baseline_v3",
            "excluded_pilots": [
                "revision_lstm_baseline_v1",
                "revision_lstm_baseline_v2",
            ],
            "best_epoch": best_epoch,
            "stopped_epoch": training_log[-1]["epoch"],
        },
    }
    (output_dir / "audit_report.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    provenance = {"version": "revision_baseline_comparison_v1", "files": {}}
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
    print(json.dumps(audit_checks, ensure_ascii=False, indent=2))
    print(report)
    print(f"OUTPUT_DIR={final_dir}")


if __name__ == "__main__":
    main()
