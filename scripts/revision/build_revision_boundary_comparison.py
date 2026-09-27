from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--boundary-weight", type=float, default=3.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"rows": len(rows), "bytes": path.stat().st_size, "sha256": digest}


def canonical_boundary_key(item: dict) -> str:
    prompt = item["messages"][1]["content"]
    prompt = re.sub(r"记录号：\d+", "记录号：<ID>", prompt)
    return json.dumps(
        {
            "case_type": item.get("case_type"),
            "prompt": prompt,
            "answer": item["messages"][-1]["content"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def normalized_prompt(item: dict) -> str:
    prompt = item["messages"][1]["content"]
    prompt = re.sub(r"记录号：\d+", "记录号：<ID>", prompt)
    prompt = re.sub(r"(边界训练样本类型|边界样本类型)：[^\n]+", "", prompt)
    prompt = re.sub(r"时间：boundary_(train|test)", "时间：<BOUNDARY>", prompt)
    return prompt


def natural_row(item: dict) -> dict:
    row = dict(item)
    row["sample_source"] = "natural"
    row["sample_weight"] = 1.0
    return row


def boundary_row(item: dict, weight: float) -> dict:
    row = dict(item)
    row["is_anomaly"] = bool(row.pop("gold"))
    row["cycle_no"] = None
    row["split"] = "train"
    row["sample_source"] = "synthetic_boundary"
    row["sample_weight"] = float(weight)
    return row


def summarize(rows: list[dict]) -> dict:
    return {
        "total": len(rows),
        "natural": sum(x["sample_source"] == "natural" for x in rows),
        "synthetic_boundary": sum(
            x["sample_source"] == "synthetic_boundary" for x in rows
        ),
        "normal": sum(not x["is_anomaly"] for x in rows),
        "anomaly": sum(x["is_anomaly"] for x in rows),
        "weight_sum": sum(float(x["sample_weight"]) for x in rows),
        "case_types": dict(
            sorted(
                Counter(
                    x.get("case_type")
                    for x in rows
                    if x["sample_source"] == "synthetic_boundary"
                ).items()
            )
        ),
    }


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    base_path = project / "data/processed/revision_grouped_v1/train_balanced.jsonl"
    old_boundary_path = project / "data/processed/boundary_train/boundary_train.jsonl"
    boundary_test_path = project / "data/processed/boundary_test/boundary_test.jsonl"
    output_dir = project / "data/processed/revision_boundary_comparison_v1"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)

    natural = [natural_row(x) for x in read_jsonl(base_path)]
    all_boundary_raw = read_jsonl(old_boundary_path)
    boundary_test = read_jsonl(boundary_test_path)

    unique_by_key: dict[str, dict] = {}
    duplicate_counts: Counter[str] = Counter()
    for item in all_boundary_raw:
        key = canonical_boundary_key(item)
        duplicate_counts[key] += 1
        unique_by_key.setdefault(key, item)
    unique_boundary_raw = list(unique_by_key.values())

    if len(all_boundary_raw) != 1017 or len(unique_boundary_raw) != 339:
        raise ValueError(
            f"Unexpected boundary counts: all={len(all_boundary_raw)}, "
            f"unique={len(unique_boundary_raw)}"
        )
    if set(duplicate_counts.values()) != {3}:
        raise ValueError("Every original boundary scenario must occur exactly 3 times")

    train_signatures = {normalized_prompt(x) for x in unique_boundary_raw}
    test_signatures = {normalized_prompt(x) for x in boundary_test}
    exact_overlap = train_signatures & test_signatures
    if exact_overlap:
        raise ValueError(
            f"Boundary train/test exact prompt overlap: {len(exact_overlap)}"
        )

    augmented = natural + [boundary_row(x, 1.0) for x in all_boundary_raw]
    weighted = natural + [
        boundary_row(x, args.boundary_weight) for x in unique_boundary_raw
    ]
    random.Random(args.seed).shuffle(augmented)
    random.Random(args.seed).shuffle(weighted)

    augmented_path = output_dir / "train_pure_augmentation.jsonl"
    weighted_path = output_dir / "train_boundary_weighted.jsonl"
    files = {
        augmented_path.name: write_jsonl(augmented_path, augmented),
        weighted_path.name: write_jsonl(weighted_path, weighted),
    }
    manifest = {
        "version": "revision_boundary_comparison_v1",
        "seed": args.seed,
        "boundary_weight": args.boundary_weight,
        "base_train": str(base_path),
        "boundary_source": str(old_boundary_path),
        "boundary_test": str(boundary_test_path),
        "boundary_source_rows": len(all_boundary_raw),
        "unique_boundary_scenarios": len(unique_boundary_raw),
        "duplicates_per_scenario": 3,
        "exact_boundary_train_test_prompt_overlap": 0,
        "design": {
            "pure_augmentation": (
                "Append all 1017 rows, i.e. three copies of each boundary scenario, "
                "with unit loss weight."
            ),
            "boundary_weighted": (
                "Append one copy of each of 339 boundary scenarios and apply "
                f"sample-level loss weight {args.boundary_weight:g}."
            ),
        },
        "summaries": {
            "pure_augmentation": summarize(augmented),
            "boundary_weighted": summarize(weighted),
        },
        "files": files,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
