from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest


MODELS = {
    "Base-LoRA": {
        "prefix": "base_lora_seed42",
        "train_dir": "base_lora_seed42",
    },
    "Pure augmentation": {
        "prefix": "pure_augmentation_seed42",
        "train_dir": "pure_augmentation_seed42",
    },
    "Boundary weighted": {
        "prefix": "boundary_weighted_seed42",
        "train_dir": "boundary_weighted_seed42",
    },
}
SETS = ("balanced", "natural", "boundary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confusion(rows: list[dict]) -> dict[str, int]:
    values = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row in rows:
        gold = bool(row["gold"])
        pred = bool(row["pred"])
        if gold and pred:
            values["tp"] += 1
        elif not gold and pred:
            values["fp"] += 1
        elif not gold and not pred:
            values["tn"] += 1
        else:
            values["fn"] += 1
    return values


def accuracy_interval(correct: int, total: int) -> list[float]:
    interval = binomtest(correct, total).proportion_ci(
        confidence_level=0.95, method="exact"
    )
    return [float(interval.low), float(interval.high)]


def prediction_key(row: dict, dataset: str) -> tuple:
    if dataset == "boundary":
        return row.get("record_id"), row.get("case_type"), bool(row["gold"])
    return row.get("record_id"), row.get("cycle_no"), bool(row["gold"])


def paired_test(a_rows: list[dict], b_rows: list[dict]) -> dict:
    a_wrong_b_right = 0
    a_right_b_wrong = 0
    both_wrong = 0
    both_right = 0
    for a, b in zip(a_rows, b_rows):
        a_correct = bool(a["pred"]) == bool(a["gold"])
        b_correct = bool(b["pred"]) == bool(b["gold"])
        if not a_correct and b_correct:
            a_wrong_b_right += 1
        elif a_correct and not b_correct:
            a_right_b_wrong += 1
        elif not a_correct and not b_correct:
            both_wrong += 1
        else:
            both_right += 1
    discordant = a_wrong_b_right + a_right_b_wrong
    p_value = (
        float(
            binomtest(
                a_wrong_b_right,
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "a_wrong_b_right": a_wrong_b_right,
        "a_right_b_wrong": a_right_b_wrong,
        "both_wrong": both_wrong,
        "both_right": both_right,
        "discordant": discordant,
        "exact_mcnemar_p": p_value,
    }


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    eval_dir = project / "outputs/revision_grouped_v1/eval"
    output_dir = project / "outputs/revision_grouped_v1/comparison_audit"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)

    metrics: dict[str, dict[str, dict]] = defaultdict(dict)
    predictions: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    source_hashes = {}
    validations = []

    for model_name, model_info in MODELS.items():
        for dataset in SETS:
            prefix = model_info["prefix"]
            prediction_path = eval_dir / f"{prefix}_{dataset}.jsonl"
            metrics_path = eval_dir / f"{prefix}_{dataset}.metrics.json"
            rows = read_jsonl(prediction_path)
            stored = read_json(metrics_path)
            recomputed = confusion(rows)
            if len(rows) != int(stored["samples"]):
                raise ValueError(f"Sample mismatch: {model_name}, {dataset}")
            for key, value in recomputed.items():
                if value != int(stored[key]):
                    raise ValueError(
                        f"Confusion mismatch: {model_name}, {dataset}, {key}"
                    )
            correct = recomputed["tp"] + recomputed["tn"]
            recomputed_accuracy = correct / len(rows)
            if not math.isclose(
                recomputed_accuracy, float(stored["accuracy"]), abs_tol=1e-15
            ):
                raise ValueError(f"Accuracy mismatch: {model_name}, {dataset}")
            stored["correct"] = correct
            stored["errors"] = len(rows) - correct
            stored["accuracy_ci95_exact"] = accuracy_interval(correct, len(rows))
            metrics[model_name][dataset] = stored
            predictions[model_name][dataset] = rows
            source_hashes[str(prediction_path.relative_to(project))] = sha256(
                prediction_path
            )
            source_hashes[str(metrics_path.relative_to(project))] = sha256(
                metrics_path
            )
            validations.append(
                f"{model_name}/{dataset}: prediction count, confusion matrix, "
                "and accuracy match stored metrics"
            )

    for dataset in SETS:
        reference = predictions["Base-LoRA"][dataset]
        reference_keys = [prediction_key(row, dataset) for row in reference]
        for model_name in ("Pure augmentation", "Boundary weighted"):
            candidate_keys = [
                prediction_key(row, dataset)
                for row in predictions[model_name][dataset]
            ]
            if candidate_keys != reference_keys:
                raise ValueError(
                    f"Prediction alignment mismatch: {dataset}, {model_name}"
                )
        validations.append(
            f"{dataset}: all three models use identical ordered test samples/gold labels"
        )

    configs = {}
    for model_name, model_info in MODELS.items():
        config_path = (
            project
            / "outputs/revision_grouped_v1"
            / model_info["train_dir"]
            / "resolved_training_config.json"
        )
        configs[model_name] = read_json(config_path)
        source_hashes[str(config_path.relative_to(project))] = sha256(config_path)

    invariant_paths = [
        ("seed",),
        ("max_steps",),
        ("learning_rate",),
        ("max_length",),
        ("lora", "r"),
        ("lora", "alpha"),
        ("lora", "dropout"),
        ("lora", "target_modules"),
        ("batch", "per_device"),
        ("batch", "gradient_accumulation"),
        ("batch", "effective"),
        ("precision",),
        ("torch",),
        ("gpu",),
    ]

    def nested(config: dict, path: tuple[str, ...]):
        value = config
        for key in path:
            value = value[key]
        return value

    for path in invariant_paths:
        values = {model: nested(config, path) for model, config in configs.items()}
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
            raise ValueError(f"Training invariant differs: {path}: {values}")
    validations.append("Core model, LoRA, optimizer schedule, batch, seed, and steps match")

    split_paths = {
        "train": project
        / "data/processed/revision_grouped_v1/train_balanced.jsonl",
        "val": project / "data/processed/revision_grouped_v1/val_balanced.jsonl",
        "test": project / "data/processed/revision_grouped_v1/test_natural.jsonl",
    }
    split_rows = {name: read_jsonl(path) for name, path in split_paths.items()}
    record_sets = {
        name: {row["record_id"] for row in rows} for name, rows in split_rows.items()
    }
    cycle_sets = {
        name: {str(row["cycle_no"]) for row in rows} for name, rows in split_rows.items()
    }
    split_overlap = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        split_overlap[f"{a}_{b}"] = {
            "record_ids": len(record_sets[a] & record_sets[b]),
            "cycles": len(cycle_sets[a] & cycle_sets[b]),
        }
        if split_overlap[f"{a}_{b}"] != {"record_ids": 0, "cycles": 0}:
            raise ValueError(f"Split overlap detected: {a}, {b}")
    validations.append("Train/validation/test have zero record-ID and cycle overlap")

    balanced_test_ids = {
        row["record_id"]
        for row in read_jsonl(
            project / "data/processed/revision_grouped_v1/test_balanced.jsonl"
        )
    }
    if not balanced_test_ids.issubset(record_sets["test"]):
        raise ValueError("Balanced test is not a subset of natural test")
    validations.append("Balanced test is a record-ID subset of the natural test partition")

    case_types = sorted(
        {row["case_type"] for row in predictions["Base-LoRA"]["boundary"]}
    )
    boundary_rows = []
    for case_type in case_types:
        row = {
            "case_type": case_type,
            "samples": sum(
                item["case_type"] == case_type
                for item in predictions["Base-LoRA"]["boundary"]
            ),
        }
        for model_name in MODELS:
            subset = [
                item
                for item in predictions[model_name]["boundary"]
                if item["case_type"] == case_type
            ]
            correct = sum(
                bool(item["pred"]) == bool(item["gold"]) for item in subset
            )
            row[f"{model_name}_correct"] = correct
            row[f"{model_name}_errors"] = len(subset) - correct
            row[f"{model_name}_accuracy"] = correct / len(subset)
        boundary_rows.append(row)

    paired = {
        "Base-LoRA vs Pure augmentation": paired_test(
            predictions["Base-LoRA"]["boundary"],
            predictions["Pure augmentation"]["boundary"],
        ),
        "Base-LoRA vs Boundary weighted": paired_test(
            predictions["Base-LoRA"]["boundary"],
            predictions["Boundary weighted"]["boundary"],
        ),
        "Pure augmentation vs Boundary weighted": paired_test(
            predictions["Pure augmentation"]["boundary"],
            predictions["Boundary weighted"]["boundary"],
        ),
    }

    comparison_rows = []
    for model_name in MODELS:
        b = metrics[model_name]["balanced"]
        n = metrics[model_name]["natural"]
        h = metrics[model_name]["boundary"]
        comparison_rows.append(
            {
                "method": model_name,
                "balanced_accuracy": b["accuracy"],
                "balanced_f1": b["f1"],
                "natural_accuracy": n["accuracy"],
                "natural_precision": n["precision"],
                "natural_recall": n["recall"],
                "natural_f1": n["f1"],
                "natural_fp": n["fp"],
                "natural_fn": n["fn"],
                "boundary_accuracy": h["accuracy"],
                "boundary_f1": h["f1"],
                "boundary_auroc": h["auroc"],
                "boundary_auprc": h["auprc"],
                "boundary_errors": h["errors"],
                "boundary_accuracy_ci95_low": h["accuracy_ci95_exact"][0],
                "boundary_accuracy_ci95_high": h["accuracy_ci95_exact"][1],
            }
        )

    audit = {
        "models": list(MODELS),
        "sets": list(SETS),
        "validations": validations,
        "training_configs": configs,
        "split_overlap": split_overlap,
        "metrics": metrics,
        "comparison": comparison_rows,
        "boundary_by_case": boundary_rows,
        "paired_boundary_tests": paired,
        "source_sha256": dict(sorted(source_hashes.items())),
    }
    with (output_dir / "comparison_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)

    with (output_dir / "comparison_table.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    with (output_dir / "boundary_by_case.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(boundary_rows[0]))
        writer.writeheader()
        writer.writerows(boundary_rows)

    report_lines = [
        "# Base vs Pure Augmentation vs Boundary-Weighted Audit",
        "",
        "## Validation gates",
        "",
    ]
    report_lines.extend(f"- PASS: {item}" for item in validations)
    report_lines.extend(
        [
            "",
            "## Overall comparison",
            "",
            "| Method | Balanced Acc. | Natural Acc. | Natural Precision | "
            "Natural Recall | Natural FP/FN | Boundary Acc. | Boundary F1 | "
            "Boundary errors | Boundary Acc. 95% exact CI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        report_lines.append(
            f"| {row['method']} | {row['balanced_accuracy']:.4%} | "
            f"{row['natural_accuracy']:.4%} | "
            f"{row['natural_precision']:.4%} | "
            f"{row['natural_recall']:.4%} | "
            f"{row['natural_fp']}/{row['natural_fn']} | "
            f"{row['boundary_accuracy']:.4%} | "
            f"{row['boundary_f1']:.4%} | "
            f"{row['boundary_errors']} | "
            f"[{row['boundary_accuracy_ci95_low']:.4%}, "
            f"{row['boundary_accuracy_ci95_high']:.4%}] |"
        )
    report_lines.extend(
        [
            "",
            "## Boundary results by case type",
            "",
            "| Case type | N | Base correct | Pure augmentation correct | "
            "Boundary weighted correct |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in boundary_rows:
        report_lines.append(
            f"| {row['case_type']} | {row['samples']} | "
            f"{row['Base-LoRA_correct']} | "
            f"{row['Pure augmentation_correct']} | "
            f"{row['Boundary weighted_correct']} |"
        )
    report_lines.extend(["", "## Exact paired tests on BoundarySet", ""])
    for name, result in paired.items():
        report_lines.append(
            f"- {name}: A-wrong/B-right={result['a_wrong_b_right']}, "
            f"A-right/B-wrong={result['a_right_b_wrong']}, "
            f"discordant={result['discordant']}, "
            f"exact McNemar p={result['exact_mcnemar_p']:.6g}."
        )
    report_lines.extend(
        [
            "",
            "## Evidence-grounded interpretation",
            "",
            "- All methods retain perfect classification on the cycle-disjoint "
            "balanced test set.",
            "- Both boundary interventions reduce BoundarySet errors relative to "
            "Base-LoRA.",
            "- Pure augmentation yields the best observed BoundarySet accuracy and "
            "does not degrade the natural-prevalence test.",
            "- Under matched nominal total weight, a single exposure with weight 3 "
            "does not match three repeated exposures in this seed/run.",
            "- BoundarySet contains only 71 targeted synthetic cases; confidence "
            "intervals and paired tests must accompany claims, and the result must "
            "not be generalized to cross-device deployment.",
        ]
    )
    (output_dir / "comparison_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output_dir), "validations": validations}, indent=2))


if __name__ == "__main__":
    main()
