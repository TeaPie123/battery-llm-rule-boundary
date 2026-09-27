from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    output = (
        project
        / "outputs/revision_grouped_v1/comparison_audit/provenance_manifest.json"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")

    paths = [
        "scripts/revision/build_revision_grouped_dataset.py",
        "scripts/revision/build_revision_boundary_comparison.py",
        "scripts/revision/train_revision_base_lora.py",
        "scripts/revision/train_revision_comparison_lora.py",
        "scripts/revision/evaluate_revision_lora.py",
        "scripts/revision/audit_revision_comparison.py",
        "data/processed/revision_grouped_v1/manifest.json",
        "data/processed/revision_grouped_v1/train_balanced.jsonl",
        "data/processed/revision_grouped_v1/val_balanced.jsonl",
        "data/processed/revision_grouped_v1/test_balanced.jsonl",
        "data/processed/revision_grouped_v1/test_natural.jsonl",
        "data/processed/revision_boundary_comparison_v1/manifest.json",
        "data/processed/revision_boundary_comparison_v1/train_pure_augmentation.jsonl",
        "data/processed/revision_boundary_comparison_v1/train_boundary_weighted.jsonl",
        "data/processed/boundary_test/boundary_test.jsonl",
        "outputs/revision_grouped_v1/base_lora_seed42/resolved_training_config.json",
        "outputs/revision_grouped_v1/base_lora_seed42/adapter_model.safetensors",
        "outputs/revision_grouped_v1/base_lora_seed42.log",
        "outputs/revision_grouped_v1/pure_augmentation_seed42/resolved_training_config.json",
        "outputs/revision_grouped_v1/pure_augmentation_seed42/adapter_model.safetensors",
        "outputs/revision_grouped_v1/pure_augmentation_seed42.log",
        "outputs/revision_grouped_v1/boundary_weighted_seed42/resolved_training_config.json",
        "outputs/revision_grouped_v1/boundary_weighted_seed42/adapter_model.safetensors",
        "outputs/revision_grouped_v1/boundary_weighted_seed42.log",
        "outputs/revision_grouped_v1/comparison_audit/comparison_audit.json",
        "outputs/revision_grouped_v1/comparison_audit/comparison_table.csv",
        "outputs/revision_grouped_v1/comparison_audit/boundary_by_case.csv",
        "outputs/revision_grouped_v1/comparison_audit/comparison_report.md",
    ]
    for prefix in (
        "base_lora_seed42",
        "pure_augmentation_seed42",
        "boundary_weighted_seed42",
    ):
        for dataset in ("balanced", "natural", "boundary"):
            paths.extend(
                [
                    f"outputs/revision_grouped_v1/eval/{prefix}_{dataset}.jsonl",
                    f"outputs/revision_grouped_v1/eval/{prefix}_{dataset}.metrics.json",
                ]
            )

    entries = {}
    for relative in paths:
        path = project / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    payload = {
        "project": str(project),
        "artifact_count": len(entries),
        "artifacts": dict(sorted(entries.items())),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "artifact_count": len(entries)}))


if __name__ == "__main__":
    main()
