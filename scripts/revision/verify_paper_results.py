"""Verify the public aggregate artifacts against the final paper tables.

This check intentionally uses only synthetic per-case outputs and aggregate
tables. It does not require the private natural-record dataset or checkpoints.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pct4(value: str | float) -> str:
    return f"{100.0 * float(value):.4f}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_table_ii() -> None:
    rows = read_csv(RESULTS / "paper_final" / "dataset_statistics.csv")
    expected = {
        "train": (548, 430170, 12381, "2.8782"),
        "validation": (117, 92133, 2653, "2.8795"),
        "test": (118, 92194, 2655, "2.8798"),
        "balanced_train": (548, 24762, 12381, "50.0000"),
        "balanced_validation": (117, 5306, 2653, "50.0000"),
        "balanced_test": (118, 5310, 2655, "50.0000"),
        "synthetic_boundary": (None, 71, 21, "29.5775"),
    }
    by_partition = {row["partition"]: row for row in rows}
    require(set(by_partition) == set(expected), "Table II partition set changed")
    for partition, (cycles, records, anomalies, prevalence) in expected.items():
        row = by_partition[partition]
        actual_cycles = int(row["cycles"]) if row["cycles"] else None
        require(actual_cycles == cycles, f"Table II cycle mismatch: {partition}")
        require(int(row["records"]) == records, f"Table II record mismatch: {partition}")
        require(int(row["anomalies"]) == anomalies, f"Table II anomaly mismatch: {partition}")
        require(pct4(row["prevalence"]) == prevalence, f"Table II prevalence mismatch: {partition}")


def verify_table_iii() -> None:
    rows = read_csv(RESULTS / "paper_final" / "cross_method_comparison.csv")
    ci_rows = read_csv(RESULTS / "paper_final" / "boundary_accuracy_exact_ci.csv")
    by_key = {(row["method"], row["dataset"]): row for row in rows}
    ci_by_method = {row["method"]: row for row in ci_rows}
    expected = {
        "Base-LoRA": ("100.0000", "99.9870", "0.0134", "0.0000", "100.0000", "99.9989", 12, 0, "61.9718", "49.6721", "73.2396", "75.5714", "44.8980"),
        "Pure augmentation": ("100.0000", "99.9989", "0.0011", "0.0000", "100.0000", "100.0000", 1, 0, "80.2817", "69.1350", "88.7792", "86.8571", "70.8333"),
        "Boundary weighted": ("100.0000", "99.9989", "0.0011", "0.0000", "100.0000", "100.0000", 1, 0, "71.8310", "59.9004", "81.8696", "78.4286", "52.3810"),
        "Isolation Forest": ("98.1733", "96.3620", "3.7414", "0.1507", "98.5531", "49.0966", 3350, 4, "36.6197", "25.4958", "48.8976", "56.6190", "45.7831"),
        "Random Forest": ("99.9812", "99.9881", "0.0123", "0.0000", "99.9988", "99.9187", 11, 0, "74.6479", "62.9240", "84.2304", "62.9524", "50.0000"),
        "LSTM": ("99.5104", "99.6345", "0.3596", "0.5650", "99.9745", "99.1049", 322, 15, "56.3380", "44.0455", "68.0850", "52.3810", "20.5128"),
        "Qwen zero-shot": ("50.0000", "2.8798", "100.0000", "0.0000", "44.2519", "2.5734", 89539, 0, "29.5775", "19.3297", "41.5933", "31.5714", "45.6522"),
        "Qwen four-shot": ("50.0000", "2.8798", "100.0000", "0.0000", "39.1817", "2.3864", 89539, 0, "29.5775", "19.3297", "41.5933", "46.1905", "45.6522"),
    }
    require(len(rows) == 24, "Table III source must contain 8 methods x 3 datasets")
    for method, values in expected.items():
        balanced = by_key[(method, "balanced")]
        natural = by_key[(method, "natural")]
        boundary = by_key[(method, "boundary")]
        ci = ci_by_method[method]
        actual = (
            pct4(balanced["accuracy"]),
            pct4(natural["accuracy"]),
            pct4(natural["fpr"]),
            pct4(natural["fnr"]),
            pct4(natural["auroc"]),
            pct4(natural["auprc"]),
            int(natural["fp"]),
            int(natural["fn"]),
            pct4(boundary["accuracy"]),
            pct4(ci["exact_95ci_lower"]),
            pct4(ci["exact_95ci_upper"]),
            pct4(boundary["auroc"]),
            pct4(boundary["f1"]),
        )
        require(actual == values, f"Table III mismatch: {method}\n{actual}\n{values}")


def verify_boundary_holm() -> None:
    raw_rows = read_csv(RESULTS / "paper_final" / "pairwise_mcnemar.csv")
    lora = {"Base-LoRA", "Pure augmentation", "Boundary weighted"}
    family = [
        row
        for row in raw_rows
        if row["dataset"] == "boundary" and {row["first"], row["second"]} <= lora
    ]
    require(len(family) == 3, "Expected three prespecified BoundarySet LoRA tests")
    ordered = sorted(family, key=lambda row: float(row["exact_two_sided_p"]))
    running = 0.0
    calculated = {}
    for rank, row in enumerate(ordered, 1):
        running = max(running, min(1.0, (4 - rank) * float(row["exact_two_sided_p"])))
        calculated[(row["first"], row["second"])] = running
    expected = {
        ("Base-LoRA", "Pure augmentation"): 0.0029296875,
        ("Base-LoRA", "Boundary weighted"): 0.03125,
        ("Pure augmentation", "Boundary weighted"): 0.14599609375,
    }
    require(calculated == expected, f"Holm values changed: {calculated}")
    published = read_csv(
        RESULTS / "paper_final" / "prespecified_boundary_lora_mcnemar_holm.csv"
    )
    for row in published:
        key = (row["first"], row["second"])
        require(
            math.isclose(float(row["holm_adjusted_p"]), expected[key], abs_tol=1e-15),
            f"Published Holm value mismatch: {key}",
        )


def verify_table_iv() -> None:
    rows = read_csv(RESULTS / "near_threshold" / "cross_method_metrics.csv")
    by_key = {(row["method"], row["rule"], row["scope"]): row for row in rows}
    for method in ["Base-LoRA", "Pure augmentation", "Boundary weighted"]:
        current = by_key[(method, "charge_current_jump", "all")]
        cell = by_key[(method, "cell_voltage_jump", "all")]
        require((int(current["samples"]), pct4(current["accuracy"]), int(current["fp"]), int(current["fn"])) == (43, "97.6744", 1, 0), f"Table IV current mismatch: {method}")
        require((int(cell["samples"]), pct4(cell["accuracy"]), int(cell["fp"]), int(cell["fn"])) == (40, "100.0000", 0, 0), f"Table IV cell mismatch: {method}")


def verify_tables_v_and_vi() -> None:
    case_path = RESULTS / "paper_final" / "representative_boundary_cases.csv"
    cases = {int(row["record_id"]): row for row in read_csv(case_path)}
    require(set(cases) == {900002, 900008, 900018, 900041, 900056}, "Table V case IDs changed")
    for prefix, filename in {
        "base": "base_lora_boundary_formal.jsonl",
        "pure": "pure_augmentation_boundary_formal.jsonl",
        "weighted": "boundary_weighted_boundary_formal.jsonl",
    }.items():
        rows = [json.loads(line) for line in (RESULTS / "paper_final" / filename).read_text(encoding="utf-8").splitlines() if line]
        require(len(rows) == 71, f"Synthetic prediction count changed: {filename}")
        by_id = {int(row["record_id"]): row for row in rows}
        require(len(by_id) == 71, f"Duplicate synthetic record ID: {filename}")
        for record_id, case in cases.items():
            row = by_id[record_id]
            require(str(bool(row["pred"])).lower() == case[f"{prefix}_pred"], f"Table V prediction mismatch: {prefix} {record_id}")
            require(math.isclose(float(row["anomaly_score"]), float(case[f"{prefix}_score"]), abs_tol=1e-15), f"Table V score mismatch: {prefix} {record_id}")

    rows = read_csv(RESULTS / "rule_verifier" / "comparison_table.csv")
    by_key = {(row["method"], row["dataset"]): row for row in rows}
    expected = {
        "Base-LoRA": (0, 12, 27),
        "Pure augmentation": (0, 1, 14),
        "Boundary weighted": (0, 1, 20),
    }
    for method, (balanced, natural, boundary) in expected.items():
        require(int(by_key[(method, "balanced")]["llm_rule_inconsistencies"]) == balanced, f"Table VI balanced mismatch: {method}")
        require(int(by_key[(method, "natural")]["llm_rule_inconsistencies"]) == natural, f"Table VI natural mismatch: {method}")
        require(int(by_key[(method, "boundary")]["llm_rule_inconsistencies"]) == boundary, f"Table VI boundary mismatch: {method}")
        for dataset in ["balanced", "natural", "boundary"]:
            row = by_key[(method, dataset)]
            require(int(row["unnecessary_overrides"]) == 0, f"Unnecessary override: {method} {dataset}")
            require(float(row["final_accuracy"]) == 1.0, f"Final consistency mismatch: {method} {dataset}")

    benchmark = {row["dataset"]: row for row in read_csv(RESULTS / "rule_verifier_benchmark" / "benchmark_summary.csv")}
    natural = benchmark["natural"]
    require(f"{float(natural['median_microseconds_per_sample']):.2f}" == "19.87", "RuleVerifier median latency changed")
    require(f"{float(natural['p95_microseconds_per_sample']):.2f}" == "19.96", "RuleVerifier p95 latency changed")


def verify_data_validity() -> None:
    values = {row["metric"]: float(row["value"]) for row in read_csv(RESULTS / "data_validity" / "voltage_redundancy.csv")}
    require(values["current_value_pairs"] == 614497, "Natural record count changed")
    require(values["current_exact_equal_rate"] == 1.0, "Voltage equality rate changed")
    sensitivity = read_csv(RESULTS / "data_validity" / "rule_removal_sensitivity.csv")
    natural = next(row for row in sensitivity if row["dataset"] == "natural_observed_all" and row["variant"] == "remove_both_total_voltage_rules")
    boundary = next(row for row in sensitivity if row["dataset"] == "synthetic_boundary" and row["variant"] == "remove_both_total_voltage_rules")
    require(int(natural["changed_vs_original"]) == 0, "Natural total-voltage sensitivity changed")
    require(int(boundary["changed_vs_original"]) == 4, "Boundary total-voltage sensitivity changed")


def main() -> None:
    verify_table_ii()
    verify_table_iii()
    verify_boundary_holm()
    verify_table_iv()
    verify_tables_v_and_vi()
    verify_data_validity()
    print("PAPER_RESULTS_CHECKS_OK")
    print("Verified Tables II-VI and the data-validity statements from public artifacts.")
    print("Note: Table V contains two presentation-rounded scores; exact values are in representative_boundary_cases.csv.")


if __name__ == "__main__":
    main()
