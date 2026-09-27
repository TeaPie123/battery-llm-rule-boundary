from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from revision_evaluate_rule_verifier_grouped import (
    DATASETS,
    gold_value,
    load_original_rule_verifier,
    read_jsonl,
    sha256,
)


REPEATS = {
    "balanced": 20,
    "natural": 10,
    "boundary": 1000,
}
WARMUP_REPEATS = 2


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def benchmark_dataset(
    dataset: str,
    user_texts: list[str],
    expected_anomalies: int,
    rule_verify,
) -> tuple[dict, list[dict]]:
    def execute_once() -> tuple[float, int]:
        started = time.perf_counter_ns()
        anomaly_count = sum(bool(rule_verify(text)[0]) for text in user_texts)
        elapsed_ns = time.perf_counter_ns() - started
        if anomaly_count != expected_anomalies:
            raise AssertionError(
                f"Benchmark output changed on {dataset}: "
                f"{anomaly_count} != {expected_anomalies}"
            )
        return elapsed_ns / 1e9, anomaly_count

    for _ in range(WARMUP_REPEATS):
        execute_once()

    raw = []
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat in range(1, REPEATS[dataset] + 1):
            elapsed, anomaly_count = execute_once()
            raw.append(
                {
                    "dataset": dataset,
                    "repeat": repeat,
                    "samples": len(user_texts),
                    "elapsed_seconds": elapsed,
                    "samples_per_second": len(user_texts) / elapsed,
                    "microseconds_per_sample": elapsed * 1e6 / len(user_texts),
                    "anomaly_count": anomaly_count,
                }
            )
    finally:
        if gc_was_enabled:
            gc.enable()

    microseconds = [row["microseconds_per_sample"] for row in raw]
    elapsed_values = [row["elapsed_seconds"] for row in raw]
    summary = {
        "dataset": dataset,
        "samples": len(user_texts),
        "anomalies": expected_anomalies,
        "warmup_repeats": WARMUP_REPEATS,
        "measured_repeats": REPEATS[dataset],
        "median_elapsed_seconds": statistics.median(elapsed_values),
        "mean_elapsed_seconds": statistics.mean(elapsed_values),
        "median_samples_per_second": (
            len(user_texts) / statistics.median(elapsed_values)
        ),
        "mean_microseconds_per_sample": statistics.mean(microseconds),
        "std_microseconds_per_sample": (
            statistics.stdev(microseconds) if len(microseconds) > 1 else 0.0
        ),
        "median_microseconds_per_sample": statistics.median(microseconds),
        "p95_microseconds_per_sample": percentile(microseconds, 0.95),
        "min_microseconds_per_sample": min(microseconds),
        "max_microseconds_per_sample": max(microseconds),
        "deterministic_output_check": "pass",
    }
    return summary, raw


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: revision_benchmark_rule_verifier.py PROJECT_DIR"
        )
    project = Path(sys.argv[1]).resolve()
    final_dir = project / "outputs/revision_rule_verifier_benchmark_v1"
    output_dir = project / "outputs/revision_rule_verifier_benchmark_v1.tmp"
    if final_dir.exists() or output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {final_dir} or {output_dir}")
    output_dir.mkdir(parents=True)

    original_source = project / "scripts/20_rule_verifier_v3.py"
    functions = load_original_rule_verifier(original_source)
    rule_verify = functions["rule_verify"]
    summaries = []
    raw_rows = []
    inputs = {}
    for dataset, relative_path in DATASETS.items():
        path = project / relative_path
        rows = read_jsonl(path)
        user_texts = [row["messages"][1]["content"] for row in rows]
        expected_anomalies = sum(gold_value(row) for row in rows)
        summary, raw = benchmark_dataset(
            dataset,
            user_texts,
            expected_anomalies,
            rule_verify,
        )
        summaries.append(summary)
        raw_rows.extend(raw)
        inputs[dataset] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "rows": len(rows),
        }

    with (output_dir / "benchmark_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "benchmark_raw.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "load_average_before_output": (
            list(os.getloadavg()) if hasattr(os, "getloadavg") else None
        ),
        "timer": "time.perf_counter_ns",
        "garbage_collection_during_measured_repeats": "disabled",
    }
    manifest = {
        "version": "revision_rule_verifier_benchmark_v1",
        "protocol": {
            "scope": (
                "Serialized user-message parsing plus deterministic five-rule "
                "execution; excludes disk I/O and model inference."
            ),
            "warmup_repeats": WARMUP_REPEATS,
            "measured_repeats": REPEATS,
            "reporting": (
                "Median and p95 microseconds per sample across complete-dataset "
                "repeats. BoundarySet uses more repeats because it has only 71 rows."
            ),
        },
        "rule_source": {
            "path": str(original_source),
            "bytes": original_source.stat().st_size,
            "sha256": sha256(original_source),
            "reuse": "AST-extracted original RuleVerifier functions",
        },
        "environment": environment,
        "inputs": inputs,
        "summaries": summaries,
        "checks": {
            "output_anomaly_count_stable_every_repeat": "pass",
            "all_datasets_benchmarked": "pass",
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# RuleVerifier latency benchmark",
        "",
        "The benchmark measures serialized user-message parsing plus deterministic "
        "five-rule execution. Disk I/O and model inference are excluded.",
        "",
        "| Dataset | Samples | Repeats | Median µs/sample | "
        "P95 µs/sample | Median samples/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset']} | {row['samples']} | "
            f"{row['measured_repeats']} | "
            f"{row['median_microseconds_per_sample']:.2f} | "
            f"{row['p95_microseconds_per_sample']:.2f} | "
            f"{row['median_samples_per_second']:.2f} |"
        )
    lines.extend(
        [
            "",
            "These CPU measurements quantify the additional deterministic "
            "verification cost in this implementation. They are environment- "
            "and serialization-dependent and must not be presented as a "
            "universal deployment latency.",
            "",
        ]
    )
    (output_dir / "benchmark_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    provenance = {
        "version": "revision_rule_verifier_benchmark_v1",
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
    print("\n".join(lines))
    print(f"OUTPUT_DIR={final_dir}")


if __name__ == "__main__":
    main()
