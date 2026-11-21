"""
@Description: Trains YOLO models (default: YOLO26-large)

Each run gets its own directory, named with the start time down to the second.
That directory keeps the training settings, one row of scores per epoch, a
performance graph, and a summary of the best epoch.
"""

import csv
import json
import os
import time
from datetime import datetime

from ultralytics import YOLO


# Ultralytics writes these columns to results.csv. The names on the left are
# written into epoch_scores.csv alongside total losses and derived val_f1.
VAL_SCORE_COLUMNS = (
    ("val_precision", "metrics/precision(B)"),
    ("val_recall", "metrics/recall(B)"),
    ("val_map_50", "metrics/mAP50(B)"),
    ("val_map_50_95", "metrics/mAP50-95(B)"),
)

EPOCH_COLUMNS = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_map_50",
    "val_map_50_95",
)

VAL_GRAPH_SERIES = (
    ("val_precision", "Val precision"),
    ("val_recall", "Val recall"),
    ("val_f1", "Val F1"),
    ("val_map_50", "Val mAP@0.5"),
)


def trainYOLO(
    dataset_dir: str,
    yolo_dir: str,
    epochs: int,
    model_weights: str = "yolo26l.pt",
    imgsz: int = 640,
    batch: int = 16,
) -> str:
    data_yaml, dataset_root = _resolve_dataset(dataset_dir)
    assert os.path.isdir(yolo_dir), f"EE: yolo dir does not exist: {yolo_dir}"
    assert epochs >= 1, f"EE: epochs must be at least 1, got {epochs}"

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    yolo_dir_abs = os.path.abspath(yolo_dir)
    output_dir = os.path.join(yolo_dir_abs, run_name)
    os.makedirs(output_dir, exist_ok=True)

    train_kwargs = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        # Absolute project path avoids Ultralytics redirecting relative paths under runs/.
        "project": yolo_dir_abs,
        "name": run_name,
        "exist_ok": True,
        "val": True,
        "save": True,
        "plots": True,
    }
    _write_json(
        os.path.join(output_dir, "training_settings.json"),
        _settings_record(run_name, model_weights, train_kwargs),
    )

    model = YOLO(model_weights)
    model.train(**train_kwargs)

    best_weights = os.path.join(output_dir, "weights", "best.pt")
    eval_model = YOLO(best_weights) if os.path.isfile(best_weights) else model
    _write_run_report(output_dir, eval_model, dataset_root, imgsz)
    return output_dir


def _resolve_dataset(dataset_dir: str) -> tuple[str, str]:
    if os.path.isfile(dataset_dir):
        data_yaml = os.path.abspath(dataset_dir)
        return data_yaml, os.path.dirname(data_yaml)
    assert os.path.isdir(dataset_dir), f"EE: dataset dir does not exist: {dataset_dir}"
    data_yaml = os.path.abspath(os.path.join(dataset_dir, "data.yaml"))
    assert os.path.isfile(data_yaml), f"EE: data.yaml not found in {dataset_dir}"
    return data_yaml, os.path.abspath(dataset_dir)


def _settings_record(run_name: str, model_weights: str, train_kwargs: dict) -> dict:
    return {
        "run_name": run_name,
        "model": model_weights,
        "train_kwargs": train_kwargs,
        "defaults_this_run_relies_on": {
            "optimizer": "auto",
            "lr0": 0.01,
            "letterbox_resize": True,
        },
        "metrics": {
            "correct_detection": (
                "A predicted bounding box is correct if it overlaps a ground-truth "
                "bounding box of the same class with an IoU of at least 0.5."
            ),
            "precision": "TP / (TP + FP). Penalises false positives.",
            "recall": "TP / (TP + FN). Penalises false negatives.",
            "f1": (
                "2 * (precision * recall) / (precision + recall). Computed per epoch "
                "from Ultralytics validation precision and recall."
            ),
            "map_50": "Mean average precision at IoU 0.5.",
            "fps": (
                "Images per second, measured once after training on single images "
                "using weights/best.pt. The network shape does not change between "
                "epochs, so this speed belongs to the run rather than to one epoch."
            ),
            "best_epoch": (
                "The epoch with the highest validation F1. Detection matters more "
                "here than tight boxes, and F1 rewards both precision and recall."
            ),
            "library_checkpoint": (
                "weights/best.pt is chosen by Ultralytics default fitness "
                "(0.1 * mAP@0.5 + 0.9 * mAP@0.5:0.95). That epoch can differ from "
                "the best F1 epoch."
            ),
            "training_scores": (
                "Ultralytics logs per-epoch training losses (box/cls/dfl or l1) but "
                "does not compute training-split precision, recall, F1, or mAP."
            ),
        },
    }


def _write_run_report(output_dir: str, model, dataset_root: str, imgsz: int) -> None:
    results_csv = _find_results_csv(output_dir)
    epochs = _epochs_from_results(results_csv) if results_csv else []
    scores_path = os.path.join(output_dir, "epoch_scores.csv")
    _write_epoch_scores(scores_path, epochs)

    graph_path = os.path.join(output_dir, "performance.png")
    graph_error = _write_performance_graph(graph_path, epochs)

    best_f1 = _best_epoch(epochs, "val_f1")
    best_map = _best_epoch(epochs, "val_map_50")
    fps = _measure_fps(model, dataset_root, imgsz)

    best_weights = os.path.join(output_dir, "weights", "best.pt")
    last_weights = os.path.join(output_dir, "weights", "last.pt")
    summary = {
        "best_epoch_by_val_f1": _epoch_snapshot(best_f1),
        "best_epoch_by_val_map_50": _epoch_snapshot(best_map),
        "fps": fps,
        "epoch_count": len(epochs),
        "files": {
            "settings": "training_settings.json",
            "epoch_scores": "epoch_scores.csv",
            "graph": None if graph_error else "performance.png",
            "library_metrics": os.path.relpath(results_csv, output_dir) if results_csv else None,
            "library_config": "args.yaml" if os.path.isfile(os.path.join(output_dir, "args.yaml")) else None,
            "best_weights": "weights/best.pt" if os.path.isfile(best_weights) else None,
            "last_weights": "weights/last.pt" if os.path.isfile(last_weights) else None,
        },
    }
    if graph_error:
        summary["graph_error"] = graph_error
    if not epochs:
        summary["note"] = "No per-epoch rows were found in results.csv."
    _write_json(os.path.join(output_dir, "run_summary.json"), summary)


def _epochs_from_results(results_csv: str) -> list[dict]:
    """Read Ultralytics results.csv into one dict per epoch.

    Sums train/*_loss and val/*_loss dynamically so both DFL-based YOLO models
    and end-to-end YOLO26 (l1_loss) produce train_loss and val_loss.
    """
    with open(results_csv, newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    epochs = []
    for raw_row in raw_rows:
        row = {key.strip(): (val.strip() if isinstance(val, str) else val) for key, val in raw_row.items() if key}
        raw_epoch = row.get("epoch")
        if raw_epoch in (None, ""):
            continue
        record: dict = {"epoch": int(float(raw_epoch))}

        train_losses = [
            float(val)
            for key, val in row.items()
            if key.startswith("train/") and key.endswith("_loss") and val not in (None, "")
        ]
        if train_losses:
            record["train_loss"] = sum(train_losses)

        val_losses = [
            float(val)
            for key, val in row.items()
            if key.startswith("val/") and key.endswith("_loss") and val not in (None, "")
        ]
        if val_losses:
            record["val_loss"] = sum(val_losses)

        for column, source in VAL_SCORE_COLUMNS:
            raw_value = row.get(source)
            if raw_value not in (None, ""):
                record[column] = float(raw_value)

        if "val_precision" in record and "val_recall" in record:
            p, r = record["val_precision"], record["val_recall"]
            record["val_f1"] = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

        epochs.append(record)

    epochs.sort(key=lambda record: record["epoch"])
    return epochs


def _write_epoch_scores(path: str, epochs: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPOCH_COLUMNS)
        writer.writeheader()
        for record in epochs:
            writer.writerow({name: record.get(name, "") for name in EPOCH_COLUMNS})


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

    # Top panel: validation detection scores at IoU 0.5
    plotted_scores = False
    for column, label in VAL_GRAPH_SERIES:
        values = [record.get(column) for record in epochs]
        if not any(value is not None for value in values):
            continue
        axes[0].plot(epoch_numbers, values, marker="o", label=label)
        plotted_scores = True
    axes[0].set_title("Validation detection scores")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    if plotted_scores:
        axes[0].legend()

    # Bottom panel: training and validation loss
    plotted_loss = False
    for column, label in (("train_loss", "Train loss"), ("val_loss", "Val loss")):
        values = [record.get(column) for record in epochs]
        if not any(value is not None for value in values):
            continue
        axes[1].plot(epoch_numbers, values, marker="o", label=label)
        plotted_loss = True
    axes[1].set_title("Total loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    if plotted_loss:
        axes[1].legend()

    figure.suptitle("YOLO performance by epoch")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return None


def _measure_fps(model, dataset_root: str, imgsz: int, image_count: int = 32, warmup: int = 3) -> dict:
    val_dir = os.path.join(dataset_root, "images", "val")
    search_root = val_dir if os.path.isdir(val_dir) else dataset_root
    images = _first_images(search_root, image_count)
    if not images:
        return {"images_per_second": None, "note": f"No images found under {search_root}."}

    try:
        import torch

        for _ in range(warmup):
            model.predict(images[0], imgsz=imgsz, verbose=False)
        _sync_cuda(torch)

        started = time.perf_counter()
        for image in images:
            model.predict(image, imgsz=imgsz, verbose=False)
        _sync_cuda(torch)
        elapsed = time.perf_counter() - started
    except Exception as exc:
        return {"images_per_second": None, "note": f"Speed measurement failed: {exc}"}

    return {
        "images_per_second": len(images) / elapsed if elapsed > 0 else None,
        "image_count": len(images),
        "warmup_images": warmup,
        "seconds": elapsed,
        "imgsz": imgsz,
        "mode": "One image at a time, including preprocessing.",
    }


def _first_images(dataset_dir: str, limit: int) -> list[str]:
    found = []
    for root, _dirs, files in os.walk(dataset_dir):
        for name in sorted(files):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                found.append(os.path.join(root, name))
                if len(found) >= limit:
                    return found
    return found


def _sync_cuda(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _find_results_csv(output_dir: str) -> str | None:
    direct = os.path.join(output_dir, "results.csv")
    if os.path.isfile(direct):
        return direct
    for root, _dirs, files in os.walk(output_dir):
        if "results.csv" in files:
            return os.path.join(root, "results.csv")
    return None


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
