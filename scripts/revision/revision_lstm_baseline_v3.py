from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import resource
import sys
import time
from collections import Counter, deque
from pathlib import Path

import ijson
import numpy as np
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
CHARGE_STATE = 110
DISCHARGE_STATE = 30
SEQUENCE_LENGTH = 8
RUN_VERSION = "revision_lstm_baseline_v3"
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
STEP_FEATURE_NAMES = [
    "state_charge",
    "state_discharge",
    "state_other_or_idle",
    "total_voltage",
    "cell_voltage",
    "current",
    "temperature",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def state_value(item: dict) -> int | None:
    value = as_number(item, "整车State状态（状态机编码）")
    return int(value) if value is not None else None


def difference(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return abs(current - previous)


def current_step(item: dict) -> np.ndarray:
    state = state_value(item)
    values = [
        float(state == CHARGE_STATE),
        float(state == DISCHARGE_STATE),
        float(state not in {CHARGE_STATE, DISCHARGE_STATE}),
        as_number(item, "动力电池内部总电压V1"),
        as_number(item, "1号电池单体电压"),
        as_number(item, "动力电池充/放电电流"),
        as_number(item, "1号温度检测点温度"),
    ]
    return np.asarray(
        [0.0 if value is None else float(value) for value in values],
        dtype=np.float32,
    )


def label_from_items(item: dict, previous: dict | None) -> bool:
    state = state_value(item)
    total_voltage = as_number(item, "动力电池内部总电压V1")
    cell_voltage = as_number(item, "1号电池单体电压")
    current = as_number(item, "动力电池充/放电电流")
    temperature = as_number(item, "1号温度检测点温度")
    total_diff = difference(
        total_voltage, as_number(previous, "动力电池内部总电压V1")
    )
    cell_diff = difference(
        cell_voltage, as_number(previous, "1号电池单体电压")
    )
    current_diff = difference(
        current, as_number(previous, "动力电池充/放电电流")
    )
    temperature_diff = difference(
        temperature, as_number(previous, "1号温度检测点温度")
    )
    return bool(
        (
            state == CHARGE_STATE
            and total_diff is not None
            and total_diff > 3.0
        )
        or (
            state == CHARGE_STATE
            and current_diff is not None
            and current_diff > 0.5
        )
        or (cell_diff is not None and cell_diff > 0.05)
        or (temperature_diff is not None and temperature_diff > 3.0)
        or (
            state == DISCHARGE_STATE
            and total_voltage is not None
            and total_voltage > 378.2
        )
    )


def padded_sequence(history: deque[np.ndarray]) -> np.ndarray:
    rows = list(history)
    if not rows:
        raise ValueError("Cannot build sequence from empty history")
    if len(rows) < SEQUENCE_LENGTH:
        rows = [rows[0]] * (SEQUENCE_LENGTH - len(rows)) + rows
    return np.stack(rows[-SEQUENCE_LENGTH:], axis=0)


def load_jsonl_id_labels(path: Path) -> dict[int, bool]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            record_id = int(item["record_id"])
            if record_id in result:
                raise ValueError(f"Duplicate record ID: {path} {record_id}")
            result[record_id] = bool(item["is_anomaly"])
    return result


def load_cycle_splits(path: Path) -> dict[str, str]:
    design = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for split, cycles in design["split_cycles"].items():
        for cycle in cycles:
            cycle = str(cycle)
            if cycle in result:
                raise ValueError(f"Duplicate cycle: {cycle}")
            result[cycle] = split
    return result


def parse_prompt_number(prompt: str, label: str, unit: str) -> float | None:
    pattern = rf"{re.escape(label)}：(?:缺失|([+-]?[0-9.]+))\s*{re.escape(unit)}"
    match = re.search(pattern, prompt)
    if not match or match.group(1) is None:
        return None
    return float(match.group(1))


def boundary_step(
    state: int,
    total_voltage: float | None,
    cell_voltage: float | None,
    current: float | None,
    temperature: float | None,
) -> np.ndarray:
    return np.asarray(
        [
            float(state == CHARGE_STATE),
            float(state == DISCHARGE_STATE),
            float(state not in {CHARGE_STATE, DISCHARGE_STATE}),
            0.0 if total_voltage is None else total_voltage,
            0.0 if cell_voltage is None else cell_voltage,
            0.0 if current is None else current,
            0.0 if temperature is None else temperature,
        ],
        dtype=np.float32,
    )


def load_boundary(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    sequences = []
    labels = []
    metadata = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item["messages"][1]["content"]
            state = int(item["state"])
            previous_step = boundary_step(
                state,
                parse_prompt_number(prompt, "上一行总电压", "V"),
                parse_prompt_number(prompt, "上一行单体电压", "V"),
                parse_prompt_number(prompt, "上一行电流", "A"),
                parse_prompt_number(prompt, "上一行温度", "℃"),
            )
            current = boundary_step(
                state,
                parse_prompt_number(prompt, "当前总电压", "V"),
                parse_prompt_number(prompt, "当前单体电压", "V"),
                parse_prompt_number(prompt, "当前电流", "A"),
                parse_prompt_number(prompt, "当前温度", "℃"),
            )
            sequence = np.stack(
                [previous_step] * (SEQUENCE_LENGTH - 1) + [current], axis=0
            )
            sequences.append(sequence)
            labels.append(bool(item["gold"]))
            metadata.append(
                {
                    "record_id": int(item["record_id"]),
                    "cycle_no": None,
                    "case_type": item.get("case_type"),
                }
            )
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels, dtype=bool),
        metadata,
    )


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        return self.head(output[:, -1, :]).squeeze(-1)


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    numerator = 2.0 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    best = np.flatnonzero(f1 == np.nanmax(f1))
    index = int(best[-1])
    return {
        "threshold": float(thresholds[index]),
        "validation_f1": float(f1[index]),
        "validation_precision": float(precision[index]),
        "validation_recall": float(recall[index]),
        "tie_break": "highest threshold among equal maximum validation F1",
    }


def compute_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
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
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def predict_scores(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[np.ndarray, float, float]:
    dataset = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    scores = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device, non_blocking=True))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_mib = 0.0
    elapsed = time.perf_counter() - started
    return np.concatenate(scores), elapsed, float(peak_mib)


def write_predictions(
    path: Path,
    metadata: list[dict],
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for meta, gold, score in zip(metadata, y_true, scores):
            row = dict(meta)
            row.update(
                {
                    "gold": bool(gold),
                    "pred": bool(score >= threshold),
                    "anomaly_score": float(score),
                }
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: revision_lstm_baseline_v3.py PROJECT_DIR")
    set_seed(SEED)
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
    final_dir = project / f"outputs/{RUN_VERSION}"
    output_dir = project / f"outputs/{RUN_VERSION}.tmp"
    if final_dir.exists() or output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {final_dir} or {output_dir}")
    output_dir.mkdir(parents=True)

    balanced_labels = {
        split: load_jsonl_id_labels(grouped_dir / f"{split}_balanced.jsonl")
        for split in ("train", "val", "test")
    }
    id_to_split = {}
    for split, labels in balanced_labels.items():
        for record_id in labels:
            if record_id in id_to_split:
                raise ValueError(f"Balanced ID in multiple splits: {record_id}")
            id_to_split[record_id] = split
    cycle_to_split = load_cycle_splits(design_path)

    sequences = {
        key: []
        for key in ["train", "val", "test_balanced", "test_natural"]
    }
    labels = {key: [] for key in sequences}
    metadata = {key: [] for key in sequences}
    history_by_cycle: dict[str, deque[np.ndarray]] = {}
    previous_by_cycle = {}
    source_count = 0
    extraction_started = time.perf_counter()

    with source.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            source_count += 1
            record_id = int(item["record_id"])
            cycle = str(item.get("cycle_no", "missing"))
            split = cycle_to_split[cycle]
            history = history_by_cycle.setdefault(
                cycle, deque(maxlen=SEQUENCE_LENGTH)
            )
            previous = previous_by_cycle.get(cycle)
            history.append(current_step(item))
            sequence = None
            label = None
            if record_id in id_to_split or split == "test":
                sequence = padded_sequence(history)
                label = label_from_items(item, previous)

            if record_id in id_to_split:
                official_split = id_to_split[record_id]
                if official_split != split:
                    raise ValueError(
                        f"Record/cycle split mismatch: {record_id}"
                    )
                if label != balanced_labels[official_split][record_id]:
                    raise ValueError(f"Label mismatch: {record_id}")
                key = (
                    "test_balanced"
                    if official_split == "test"
                    else official_split
                )
                sequences[key].append(sequence)
                labels[key].append(label)
                metadata[key].append(
                    {
                        "record_id": record_id,
                        "cycle_no": cycle,
                        "case_type": None,
                    }
                )
            if split == "test":
                sequences["test_natural"].append(sequence)
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
    for key in sequences:
        arrays[key] = (
            np.asarray(sequences[key], dtype=np.float32),
            np.asarray(labels[key], dtype=bool),
            metadata[key],
        )
    del sequences, labels
    expected = {
        "train": 24762,
        "val": 5306,
        "test_balanced": 5310,
        "test_natural": 92194,
    }
    for key, count in expected.items():
        if arrays[key][0].shape != (
            count,
            SEQUENCE_LENGTH,
            len(STEP_FEATURE_NAMES),
        ):
            raise ValueError(f"Sequence shape mismatch: {key} {arrays[key][0].shape}")

    x_boundary, y_boundary, meta_boundary = load_boundary(boundary_path)
    train_x, train_y, _ = arrays["train"]
    val_x, val_y, _ = arrays["val"]
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1.0

    def normalize(x: np.ndarray) -> np.ndarray:
        return ((x - mean) / std).astype(np.float32)

    for key in arrays:
        arrays[key] = (normalize(arrays[key][0]), arrays[key][1], arrays[key][2])
    x_boundary = normalize(x_boundary)
    train_x, train_y, _ = arrays["train"]
    val_x, val_y, _ = arrays["val"]

    np.savez_compressed(
        output_dir / "sequence_data.npz",
        step_feature_names=np.asarray(STEP_FEATURE_NAMES),
        sequence_length=np.asarray([SEQUENCE_LENGTH]),
        normalization_mean=mean,
        normalization_std=std,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_balanced_x=arrays["test_balanced"][0],
        test_balanced_y=arrays["test_balanced"][1],
        test_natural_x=arrays["test_natural"][0],
        test_natural_y=arrays["test_natural"][1],
        boundary_x=x_boundary,
        boundary_y=y_boundary,
    )
    extraction_seconds = time.perf_counter() - extraction_started

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(len(STEP_FEATURE_NAMES), hidden_size=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_y.astype(np.float32)),
        ),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(val_x),
            torch.from_numpy(val_y.astype(np.float32)),
        ),
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    best_val_loss = math.inf
    best_epoch = 0
    patience = EARLY_STOPPING_PATIENCE
    epochs_without_improvement = 0
    training_rows = []
    training_started = time.perf_counter()
    checkpoint_path = output_dir / "lstm_best.pt"
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * batch_y.size(0)
            train_count += batch_y.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss_sum += float(loss.detach().cpu()) * batch_y.size(0)
                val_count += batch_y.size(0)
        train_loss = train_loss_sum / train_count
        val_loss = val_loss_sum / val_count
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        training_rows.append(row)
        with (output_dir / "training_log.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(row) + "\n")

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": len(STEP_FEATURE_NAMES),
                    "hidden_size": 64,
                    "sequence_length": SEQUENCE_LENGTH,
                    "normalization_mean": mean,
                    "normalization_std": std,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_scores, _, _ = predict_scores(model, val_x, device)
    threshold_info = tune_threshold(val_y, val_scores)
    checkpoint["threshold_selection"] = threshold_info
    torch.save(checkpoint, checkpoint_path)

    dataset_map = {
        "balanced": arrays["test_balanced"],
        "natural": arrays["test_natural"],
        "boundary": (x_boundary, y_boundary, meta_boundary),
    }
    metrics_by_dataset = {}
    comparison_rows = []
    for dataset_name, (x_test, y_test, rows) in dataset_map.items():
        scores, elapsed, peak_mib = predict_scores(model, x_test, device)
        current_metrics = compute_metrics(
            y_test, scores, threshold_info["threshold"]
        )
        current_metrics.update(
            {
                "elapsed_seconds": elapsed,
                "samples_per_second": len(y_test) / elapsed,
                "peak_gpu_memory_mib": peak_mib,
            }
        )
        metrics_by_dataset[dataset_name] = current_metrics
        write_predictions(
            output_dir / f"lstm_{dataset_name}.jsonl",
            rows,
            y_test,
            scores,
            threshold_info["threshold"],
        )
        (output_dir / f"lstm_{dataset_name}.metrics.json").write_text(
            json.dumps(current_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        comparison_rows.append({"model": "LSTM", "dataset": dataset_name, **current_metrics})

    with (output_dir / "comparison_table.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    manifest = {
        "version": RUN_VERSION,
        "seed": SEED,
        "device": str(device),
        "source": str(source),
        "split_design": str(design_path),
        "boundary_set": str(boundary_path),
        "sequence": {
            "length": SEQUENCE_LENGTH,
            "step_feature_names": STEP_FEATURE_NAMES,
            "natural_padding": (
                "Left-pad with the first available within-cycle observation."
            ),
            "boundary_padding": (
                "BoundarySet exposes only previous/current observations; use "
                "seven copies of previous followed by current."
            ),
        },
        "counts": {
            "source": source_count,
            **{key: int(value[0].shape[0]) for key, value in arrays.items()},
            "boundary": int(x_boundary.shape[0]),
        },
        "model": {
            "architecture": "one-layer LSTM(64) + Linear(64,32) + ReLU + Dropout(0.1) + Linear(32,1)",
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "loss": "BCEWithLogitsLoss",
            "batch_size": 256,
            "max_epochs": MAX_EPOCHS,
            "early_stopping": (
                f"validation loss, patience={EARLY_STOPPING_PATIENCE}"
            ),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "threshold_selection": threshold_info,
        },
        "timing": {
            "feature_extraction_seconds": extraction_seconds,
            "training_seconds": training_seconds,
        },
        "metrics": metrics_by_dataset,
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "peak_process_memory_mib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    provenance = {"version": RUN_VERSION, "files": {}}
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
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"OUTPUT_DIR={final_dir}")


if __name__ == "__main__":
    main()
