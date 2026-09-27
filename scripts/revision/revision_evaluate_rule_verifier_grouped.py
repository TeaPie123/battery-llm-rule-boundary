from __future__ import annotations

import ast
import csv
import hashlib
import json
import resource
import sys
import time
from pathlib import Path


METHODS = {
    "Base-LoRA": "base_lora_seed42",
    "Pure augmentation": "pure_augmentation_seed42",
    "Boundary weighted": "boundary_weighted_seed42",
}
DATASETS = {
    "balanced": "data/processed/revision_grouped_v1/test_balanced.jsonl",
    "natural": "data/processed/revision_grouped_v1/test_natural.jsonl",
    "boundary": "data/processed/boundary_test/boundary_test.jsonl",
}
REQUIRED_RULE_FUNCTIONS = {
    "first_float",
    "first_int",
    "parse_record_id",
    "abs_diff",
    "rule_verify",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_text(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Not a Boolean value: {value!r}")


def load_original_rule_verifier(source_path: Path):
    """Compile only the original parser/verifier functions, without top-level I/O."""
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in REQUIRED_RULE_FUNCTIONS
    ]
    names = {node.name for node in selected}
    if names != REQUIRED_RULE_FUNCTIONS:
        raise AssertionError(
            f"Original RuleVerifier functions missing: "
            f"{sorted(REQUIRED_RULE_FUNCTIONS - names)}"
        )
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    import re

    namespace["re"] = re
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


def gold_value(row: dict) -> bool:
    if "is_anomaly" in row:
        return bool(row["is_anomaly"])
    if "gold" in row:
        return bool(row["gold"])
    raise KeyError("Reference row lacks is_anomaly/gold")


def confusion(gold: list[bool], pred: list[bool]) -> dict:
    if len(gold) != len(pred):
        raise AssertionError("Metric length mismatch")
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    tn = sum((not g) and (not p) for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    samples = len(gold)
    accuracy = (tp + tn) / samples if samples else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    return {
        "samples": samples,
        "prevalence": sum(gold) / samples if samples else 0.0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def verify_old_regression(
    project: Path, rule_verify, parse_record_id
) -> dict:
    test_path = project / "data/processed/lora_full_balanced_v3/test.jsonl"
    prediction_path = (
        project
        / "outputs/qwen_lora_v3_full_balanced/"
        "eval_full_batched_predictions.jsonl"
    )
    archived_path = project / "outputs/rule_verifier/rule_verifier_cases.csv"
    for path in [test_path, prediction_path, archived_path]:
        if not path.exists():
            raise FileNotFoundError(f"Old RuleVerifier regression input missing: {path}")

    references = {}
    for row in read_jsonl(test_path):
        record_id = parse_record_id(row["messages"][1]["content"])
        if record_id is None:
            raise AssertionError("Could not parse old test record ID")
        references[int(record_id)] = row
    predictions = {
        int(row["record_id"]): row for row in read_jsonl(prediction_path)
    }
    with archived_path.open("r", encoding="utf-8-sig", newline="") as handle:
        archived_rows = list(csv.DictReader(handle))
    archived = {int(row["record_id"]): row for row in archived_rows}
    if set(references) != set(predictions) or set(references) != set(archived):
        raise AssertionError("Old RuleVerifier regression ID sets do not match")

    mismatches = []
    for record_id, reference in references.items():
        rule_pred, _, _ = rule_verify(reference["messages"][1]["content"])
        llm_pred = bool(predictions[record_id]["pred"])
        final_pred = rule_pred if llm_pred != rule_pred else llm_pred
        old = archived[record_id]
        expected = {
            "rule_pred": bool_text(old["rule_pred"]),
            "llm_pred": bool_text(old["llm_pred"]),
            "final_pred": bool_text(old["final_pred"]),
        }
        actual = {
            "rule_pred": bool(rule_pred),
            "llm_pred": llm_pred,
            "final_pred": bool(final_pred),
        }
        if actual != expected:
            mismatches.append(
                {"record_id": record_id, "expected": expected, "actual": actual}
            )
    if mismatches:
        raise AssertionError(
            f"Original RuleVerifier regression mismatches: {mismatches[:3]}"
        )
    corrections = sum(
        bool_text(row["llm_pred"]) != bool_text(row["rule_pred"])
        for row in archived_rows
    )
    if len(archived_rows) != 3695 or corrections != 3:
        raise AssertionError(
            f"Unexpected archived RuleVerifier counts: "
            f"rows={len(archived_rows)} corrections={corrections}"
        )
    return {
        "status": "pass",
        "samples": len(archived_rows),
        "per_record_mismatches": 0,
        "archived_corrections": corrections,
        "test_sha256": sha256(test_path),
        "prediction_sha256": sha256(prediction_path),
        "archived_cases_sha256": sha256(archived_path),
    }


def evaluate_rule_dataset(
    dataset: str, rows: list[dict], rule_verify
) -> tuple[list[dict], dict]:
    started = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    output = []
    parse_missing = {
        key: 0
        for key in [
            "state",
            "current_total_voltage",
            "total_voltage_diff",
            "cell_voltage_diff",
            "current_diff",
            "temperature_diff",
        ]
    }
    for row in rows:
        user_text = row["messages"][1]["content"]
        rule_pred, rule_reason, parsed = rule_verify(user_text)
        gold = gold_value(row)
        for key in parse_missing:
            parse_missing[key] += int(parsed[key] is None)
        output.append(
            {
                "record_id": int(row["record_id"]),
                "cycle_no": row.get("cycle_no"),
                "case_type": row.get("case_type"),
                "dataset": dataset,
                "gold": gold,
                "rule_pred": bool(rule_pred),
                "matches_gold": bool(rule_pred) == gold,
                "rule_reason": rule_reason,
                **parsed,
            }
        )
    elapsed = time.perf_counter() - started
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mismatches = [row for row in output if not row["matches_gold"]]
    if mismatches:
        raise AssertionError(
            f"RuleVerifier/gold mismatches on {dataset}: {mismatches[:3]}"
        )
    record_ids = [row["record_id"] for row in output]
    if len(record_ids) != len(set(record_ids)):
        raise AssertionError(f"Duplicate reference record IDs: {dataset}")
    metrics = confusion(
        [row["gold"] for row in output],
        [row["rule_pred"] for row in output],
    )
    return output, {
        **metrics,
        "gold_mismatches": 0,
        "parse_missing_counts": parse_missing,
        "elapsed_seconds": elapsed,
        "samples_per_second": len(output) / elapsed,
        "microseconds_per_sample": elapsed * 1e6 / len(output),
        "process_peak_rss_kib_before": rss_before,
        "process_peak_rss_kib_after": rss_after,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: revision_evaluate_rule_verifier_grouped.py PROJECT_DIR"
        )
    project = Path(sys.argv[1]).resolve()
    final_dir = project / "outputs/revision_rule_verifier_grouped_v1"
    output_dir = project / "outputs/revision_rule_verifier_grouped_v1.tmp"
    if final_dir.exists() or output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {final_dir} or {output_dir}")
    output_dir.mkdir(parents=True)

    original_source = project / "scripts/20_rule_verifier_v3.py"
    original_functions = load_original_rule_verifier(original_source)
    rule_verify = original_functions["rule_verify"]
    old_regression = verify_old_regression(
        project, rule_verify, original_functions["parse_record_id"]
    )

    references: dict[str, list[dict]] = {}
    reference_by_id: dict[str, dict[int, dict]] = {}
    rule_outputs: dict[str, list[dict]] = {}
    rule_metrics: dict[str, dict] = {}
    input_manifest: dict[str, dict] = {
        "original_rule_verifier_source": {
            "path": str(original_source),
            "bytes": original_source.stat().st_size,
            "sha256": sha256(original_source),
        }
    }
    for dataset, relative_path in DATASETS.items():
        path = project / relative_path
        rows = read_jsonl(path)
        references[dataset] = rows
        reference_by_id[dataset] = {
            int(row["record_id"]): row for row in rows
        }
        if len(reference_by_id[dataset]) != len(rows):
            raise AssertionError(f"Duplicate reference IDs: {dataset}")
        outputs, metrics = evaluate_rule_dataset(dataset, rows, rule_verify)
        rule_outputs[dataset] = outputs
        rule_metrics[dataset] = metrics
        input_manifest[f"reference_{dataset}"] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "rows": len(rows),
        }
        with (output_dir / f"rule_predictions_{dataset}.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in outputs:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    comparison_rows = []
    inconsistent_rows = []
    for method, prefix in METHODS.items():
        for dataset in DATASETS:
            prediction_path = (
                project
                / "outputs/revision_grouped_v1/eval"
                / f"{prefix}_{dataset}.jsonl"
            )
            predictions = read_jsonl(prediction_path)
            prediction_by_id = {
                int(row["record_id"]): row for row in predictions
            }
            if len(prediction_by_id) != len(predictions):
                raise AssertionError(
                    f"Duplicate prediction IDs: {method} {dataset}"
                )
            official = reference_by_id[dataset]
            if set(prediction_by_id) != set(official):
                raise AssertionError(
                    f"Prediction/reference ID mismatch: {method} {dataset}"
                )
            gold = []
            llm_pred = []
            rule_pred = []
            corrected_model_errors = 0
            unnecessary_overrides = 0
            for rule_row in rule_outputs[dataset]:
                record_id = rule_row["record_id"]
                prediction = prediction_by_id[record_id]
                official_gold = gold_value(official[record_id])
                if bool(prediction["gold"]) != official_gold:
                    raise AssertionError(
                        f"Prediction gold mismatch: {method} {dataset} {record_id}"
                    )
                model_value = bool(prediction["pred"])
                verifier_value = bool(rule_row["rule_pred"])
                if model_value != verifier_value:
                    corrected_model_errors += int(model_value != official_gold)
                    unnecessary_overrides += int(model_value == official_gold)
                    inconsistent_rows.append(
                        {
                            "method": method,
                            "dataset": dataset,
                            "record_id": record_id,
                            "cycle_no": rule_row.get("cycle_no"),
                            "case_type": rule_row.get("case_type"),
                            "gold": official_gold,
                            "llm_pred": model_value,
                            "rule_pred": verifier_value,
                            "final_pred": verifier_value,
                            "anomaly_score": prediction.get("anomaly_score"),
                            "rule_reason": rule_row["rule_reason"],
                        }
                    )
                gold.append(official_gold)
                llm_pred.append(model_value)
                rule_pred.append(verifier_value)
            llm_metrics = confusion(gold, llm_pred)
            final_metrics = confusion(gold, rule_pred)
            inconsistencies = sum(
                first != second
                for first, second in zip(llm_pred, rule_pred)
            )
            if corrected_model_errors != inconsistencies:
                raise AssertionError(
                    f"Not every override corrects a model error: {method} {dataset}"
                )
            if unnecessary_overrides:
                raise AssertionError(
                    f"RuleVerifier overrides correct model labels: {method} {dataset}"
                )
            if final_metrics["accuracy"] != 1.0:
                raise AssertionError(
                    f"Final rule-consistent output is not exact: {method} {dataset}"
                )
            comparison_rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "samples": len(gold),
                    "prevalence": sum(gold) / len(gold),
                    "llm_accuracy": llm_metrics["accuracy"],
                    "llm_fp": llm_metrics["fp"],
                    "llm_fn": llm_metrics["fn"],
                    "llm_rule_inconsistencies": inconsistencies,
                    "corrected_model_errors": corrected_model_errors,
                    "unnecessary_overrides": unnecessary_overrides,
                    "final_accuracy": final_metrics["accuracy"],
                    "final_tp": final_metrics["tp"],
                    "final_fp": final_metrics["fp"],
                    "final_tn": final_metrics["tn"],
                    "final_fn": final_metrics["fn"],
                }
            )
            input_manifest[f"prediction_{prefix}_{dataset}"] = {
                "path": str(prediction_path),
                "bytes": prediction_path.stat().st_size,
                "sha256": sha256(prediction_path),
                "rows": len(predictions),
            }

    comparison_fields = list(comparison_rows[0].keys())
    with (output_dir / "comparison_table.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    with (output_dir / "inconsistent_cases.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in inconsistent_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_key = {
        (row["method"], row["dataset"]): row for row in comparison_rows
    }
    report_lines = [
        "# RuleVerifier evaluation on cycle-isolated revision datasets",
        "",
        "## Audit status",
        "",
        "- The original `rule_verify` function was extracted by AST from "
        "`scripts/20_rule_verifier_v3.py`; no replacement rule implementation "
        "was introduced.",
        f"- Old-result regression: {old_regression['samples']} records matched "
        "the archived per-record outputs with zero mismatches; "
        f"{old_regression['archived_corrections']} archived corrections reproduced.",
        "- RuleVerifier predictions matched the executable-rule gold labels for "
        "every balanced, natural, and BoundarySet record.",
        "- Consequently, final accuracy is 100% by construction under the same "
        "rule specification. This is consistency enforcement, not independent "
        "evidence of real-world diagnostic accuracy.",
        "",
        "## Natural-test corrections",
        "",
        "| Method | Samples | LLM accuracy | LLM FP/FN | "
        "LLM–rule inconsistencies | Final accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = by_key[(method, "natural")]
        report_lines.append(
            f"| {method} | {row['samples']} | {row['llm_accuracy']:.4%} | "
            f"{row['llm_fp']}/{row['llm_fn']} | "
            f"{row['llm_rule_inconsistencies']} | "
            f"{row['final_accuracy']:.4%} |"
        )
    report_lines.extend(
        [
            "",
            "## RuleVerifier latency",
            "",
            "| Dataset | Samples | Elapsed (s) | Samples/s | µs/sample |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        metric = rule_metrics[dataset]
        report_lines.append(
            f"| {dataset} | {metric['samples']} | "
            f"{metric['elapsed_seconds']:.6f} | "
            f"{metric['samples_per_second']:.2f} | "
            f"{metric['microseconds_per_sample']:.2f} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "RuleVerifier re-executes the same five-rule specification used to "
            "define the labels. Its value is deterministic consistency "
            "enforcement and correction accounting. It is not an independent "
            "learned baseline, it cannot detect anomalies omitted from the "
            "rules, and its 100% rule-label agreement must not be presented as "
            "external industrial validation.",
            "",
        ]
    )
    (output_dir / "comparison_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    manifest = {
        "version": "revision_rule_verifier_grouped_v1",
        "rule_implementation": {
            "reuse_policy": (
                "AST extraction of the original first_float, first_int, "
                "parse_record_id, abs_diff, and rule_verify functions; "
                "top-level legacy I/O is not executed."
            ),
            "source_path": str(original_source),
            "source_sha256": sha256(original_source),
        },
        "old_result_regression": old_regression,
        "dataset_rule_metrics": rule_metrics,
        "methods": METHODS,
        "comparison_rows": comparison_rows,
        "checks": {
            "original_regression": "pass",
            "reference_id_uniqueness": "pass",
            "prediction_id_alignment": "pass",
            "prediction_gold_alignment": "pass",
            "rule_prediction_equals_gold_all_records": "pass",
            "all_overrides_correct_model_errors": "pass",
            "unnecessary_overrides": 0,
        },
        "inputs": input_manifest,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    provenance = {
        "version": "revision_rule_verifier_grouped_v1",
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
    print("\n".join(report_lines))
    print(f"OUTPUT_DIR={final_dir}")


if __name__ == "__main__":
    main()
