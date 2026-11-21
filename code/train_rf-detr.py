"""
@Description: Trains rf-detr models

Each run gets its own directory, named with the start time down to the second.
That directory keeps the training settings, one row of scores per epoch, a
performance graph, and a summary of the best epoch.
"""

import csv
import json
import os
import time
from datetime import datetime

from rfdetr import RFDETRBase


# Lightning logs these names. The CSV columns on the left are the names written
# into epoch_scores.csv for the graph.
EPOCH_METRICS = (
    ("train_loss", "train/loss"),
    ("train_precision", "train/precision"),
    ("train_recall", "train/recall"),
    ("train_f1", "train/F1"),
    ("train_map_50", "train/mAP_50"),
    ("val_loss", "val/loss"),
    ("val_precision", "val/precision"),
    ("val_recall", "val/recall"),
    ("val_f1", "val/F1"),
    ("val_map_50", "val/mAP_50"),
    ("val_map_50_95", "val/mAP_50_95"),
    ("val_mar", "val/mAR"),
)

GRAPH_SERIES = (
    ("val_precision", "Val precision"),
    ("val_recall", "Val recall"),
    ("val_f1", "Val F1"),
    ("val_map_50", "Val mAP@0.5"),
    ("train_precision", "Train precision"),
    ("train_recall", "Train recall"),
    ("train_f1", "Train F1"),
    ("train_map_50", "Train mAP@0.5"),
)


def trainRFDETR(dataset_dir: str, rf_detr_dir: str, epochs: int):
    assert os.path.isdir(dataset_dir), f"EE: dataset dir does not exist: {dataset_dir}"
    assert os.path.isdir(rf_detr_dir), f"EE: rf detr dir does not exist: {rf_detr_dir}"
    assert epochs >= 1, f"EE: epochs must be at least 1, got {epochs}"

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(rf_detr_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    train_kwargs = {
        "dataset_dir": dataset_dir,
        "epochs": epochs,
        "batch_size": 4,
        "grad_accum_steps": 4,
        "lr": 1e-4,
        "output_dir": output_dir,
        # Score every epoch so the graph has a point for each one.
        "eval_interval": 1,
        # Train precision, recall, F1, and mAP, not only the training loss.
        "compute_train_metrics": True,
        "compute_val_loss": True,
        "tensorboard": True,
    }
    _write_json(os.path.join(output_dir, "training_settings.json"), _settings_record(run_name, train_kwargs))

    model = RFDETRBase()
    model.train(**train_kwargs)
    _write_run_report(output_dir, model, dataset_dir)
    return output_dir


def _settings_record(run_name: str, train_kwargs: dict) -> dict:
    return {
        "run_name": run_name,
        "model": "RFDETRBase",
        "train_kwargs": train_kwargs,
        "defaults_this_run_relies_on": {
            "use_ema": True,
            "best_model_metric": "map",
            "checkpoint_interval": 10,
        },
        "metrics": {
            "correct_detection": (
                "A predicted bounding box is correct if it overlaps a ground-truth "
                "bounding box of the same class with an IoU of at least 0.5."
            ),
            "precision": "TP / (TP + FP). Penalises false positives.",
            "recall": "TP / (TP + FN). Penalises false negatives.",
            "f1": "2 * (precision * recall) / (precision + recall).",
            "map_50": "Mean average precision at IoU 0.5.",
            "fps": (
                "Images per second, measured once after training on single images. "
                "The network shape does not change between epochs, so this speed "
                "belongs to the run rather than to one epoch."
            ),
            "best_epoch": (
                "The epoch with the highest validation F1. Detection matters more "
                "here than tight boxes, and F1 rewards both precision and recall."
            ),
            "library_checkpoint": (
                "checkpoint_best_total.pth is chosen by validation mAP@0.5:0.95, "
                "which is RF-DETR's default. That epoch can differ from the best F1 epoch."
            ),
            "validation_weights": "Validation scores are from the EMA weights.",
            "training_scores": (
                "Training precision, recall, F1, and mAP are computed on the augmented "
                "training images, from the live weights rather than the EMA weights."
            ),
        },
    }


def _write_run_report(output_dir: str, model, dataset_dir: str) -> None:
    metrics_csv = _find_metrics_csv(output_dir)
    epochs = _epochs_from_metrics(metrics_csv) if metrics_csv else []
    scores_path = os.path.join(output_dir, "epoch_scores.csv")
    _write_epoch_scores(scores_path, epochs)

    graph_path = os.path.join(output_dir, "performance.png")
    graph_error = _write_performance_graph(graph_path, epochs)

    best_f1 = _best_epoch(epochs, "val_f1")
    best_map = _best_epoch(epochs, "val_map_50")
    fps = _measure_fps(model, dataset_dir)

    summary = {
        "best_epoch_by_val_f1": _epoch_snapshot(best_f1),
        "best_epoch_by_val_map_50": _epoch_snapshot(best_map),
        "fps": fps,
        "epoch_count": len(epochs),
        "files": {
            "settings": "training_settings.json",
            "epoch_scores": "epoch_scores.csv",
            "graph": None if graph_error else "performance.png",
            "library_metrics": os.path.relpath(metrics_csv, output_dir) if metrics_csv else None,
            "library_config": "training_config.json",
        },
    }
    if graph_error:
        summary["graph_error"] = graph_error
    if not epochs:
        summary["note"] = "No per-epoch rows were found in metrics.csv."
    _write_json(os.path.join(output_dir, "run_summary.json"), summary)


def _epochs_from_metrics(metrics_csv: str) -> list[dict]:
    """Collapse Lightning's metrics.csv to one dict per epoch.

    Lightning writes a row each time it logs, so one epoch spans several rows
    and most cells are blank. The last value seen for a column is the epoch
    aggregate, because epoch metrics are logged after the step metrics.
    """
    with open(metrics_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_epoch: dict[int, dict] = {}
    for row in rows:
        raw_epoch = row.get("epoch")
        if raw_epoch in (None, ""):
            continue
        epoch = int(float(raw_epoch)) + 1
        record = by_epoch.setdefault(epoch, {"epoch": epoch})
        for column, source in EPOCH_METRICS:
            raw_value = row.get(source)
            if raw_value not in (None, ""):
                record[column] = float(raw_value)
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _write_epoch_scores(path: str, epochs: list[dict]) -> None:
    fieldnames = ["epoch", *[column for column, _source in EPOCH_METRICS]]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in epochs:
            writer.writerow({name: record.get(name, "") for name in fieldnames})


def _best_epoch(epochs: list[dict], metric: str) -> dict | None:
    scored = [record for record in epochs if metric in record]
    if not scored:
        return None
    return max(scored, key=lambda record: (record[metric], record["epoch"]))


def _epoch_snapshot(record: dict | None) -> dict | None:
    if record is None:
        return None
    validation = _score_group(record, "val_")
    training = _score_group(record, "train_")
    return {"epoch": record["epoch"], "validation": validation, "training": training}


def _score_group(record: dict, prefix: str) -> dict:
    return {key[len(prefix) :]: value for key, value in record.items() if key.startswith(prefix)}


def _write_performance_graph(path: str, epochs: list[dict]) -> str | None:
    if not epochs:
        return "No epoch scores to plot."
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return f"matplotlib is not available: {exc}"

    epoch_numbers = [record["epoch"] for record in epochs]
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    panels = (
        (axes[0], "Validation", [item for item in GRAPH_SERIES if item[0].startswith("val_")]),
        (axes[1], "Training", [item for item in GRAPH_SERIES if item[0].startswith("train_")]),
    )
    for axis, title, series in panels:
        plotted = False
        for column, label in series:
            values = [record.get(column) for record in epochs]
            if not any(value is not None for value in values):
                continue
            axis.plot(epoch_numbers, values, marker="o", label=label)
            plotted = True
        axis.set_title(title)
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1)
        axis.grid(True, alpha=0.3)
        if plotted:
            axis.legend()
    axes[1].set_xlabel("Epoch")
    figure.suptitle("Detection scores at IoU 0.5")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return None


def _measure_fps(model, dataset_dir: str, image_count: int = 32, warmup: int = 3) -> dict:
    images = _first_images(dataset_dir, image_count)
    if not images:
        return {"images_per_second": None, "note": f"No images found under {dataset_dir}."}

    try:
        import torch

        for _ in range(warmup):
            model.predict(images[0], include_source_image=False)
        _sync_cuda(torch)

        started = time.perf_counter()
        for image in images:
            model.predict(image, include_source_image=False)
        _sync_cuda(torch)
        elapsed = time.perf_counter() - started
    except Exception as exc:
        return {"images_per_second": None, "note": f"Speed measurement failed: {exc}"}

    return {
        "images_per_second": len(images) / elapsed if elapsed > 0 else None,
        "image_count": len(images),
        "warmup_images": warmup,
        "seconds": elapsed,
        "mode": "One image at a time, including preprocessing.",
    }


def _first_images(dataset_dir: str, limit: int) -> list[str]:
    found = []
    for root, _dirs, files in os.walk(dataset_dir):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                found.append(os.path.join(root, name))
                if len(found) >= limit:
                    return found
    return found


def _sync_cuda(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _find_metrics_csv(output_dir: str) -> str | None:
    direct = os.path.join(output_dir, "metrics.csv")
    if os.path.isfile(direct):
        return direct
    for root, _dirs, files in os.walk(output_dir):
        if "metrics.csv" in files:
            return os.path.join(root, "metrics.csv")
    return None


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
