"""
@Description: Converts the UAVDT download (Supervisely format, from Dataset Ninja)
into YOLO format for Ultralytics and COCO format for RF-DETR.

@Notes:
- Source layout: <src>/{train,test}/{img,ann}/, one <image>.json per image, and
  <src>/meta.json with the class ids. Boxes are absolute pixel corners
  [[x1, y1], [x2, y2]].
- Only the "M" sequences are kept. They are the UAVDT detection benchmark, with
  every vehicle labelled car, truck, or bus. The "S" sequences in the test split
  are the single-object tracking benchmark: one target per frame is labelled, as
  the generic class "vehicle", and every other car is unlabelled, so correct
  detections there would be scored as false positives.
- The validation split is taken from train by whole sequence, so frames from
  one video never land in both train and val.
- Images are hard linked into each output, not copied, so the outputs take
  almost no extra disk space. If hard links are not possible (another drive),
  the images are copied. Hard links are ordinary files, so the folders can be
  zipped and uploaded to Colab as they are.

Outputs:
    <dst>/uavdt_yolo/
        data.yaml
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt      class cx cy w h, normalized 0-1
    <dst>/uavdt_coco/
        {train,valid,test}/*.jpg + _annotations.coco.json
"""

import json
import os
import random
import shutil
from collections import Counter, defaultdict

# Output class order. YOLO class ids are the list index; COCO category ids are index + 1.
CLASSES = ("car", "truck", "bus")

# Source split each output split is drawn from.
SOURCE_SPLIT = {"train": "train", "val": "train", "test": "test"}

# RF-DETR's folder name for each split.
COCO_SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}

DETECTION_SEQUENCE_PREFIX = "M"

# M-sequences with severely broken or corrupted annotations in the Dataset Ninja export:
# - M0201 (train): missed large vehicles and a shifted box on empty road
# - M0207 (train): only 0-3 vehicles boxed in a busy intersection
# - M1006 (train): bogus truck box over an empty sidewalk and missed edge vehicles
# - M0208 (test): bridge over boats with only 2-5 boxes and missed cars
# - M0802, M1007, M1101 (test): giant ignore-region boxes mislabeled as "car"
EXCLUDED_SEQUENCES = frozenset({
    "M0201",
    "M0207",
    "M1006",
    "M0208",
    "M0802",
    "M1007",
    "M1101",
})


def prepareUAVDT(
    src_dir: str,
    dst_dir: str,
    val_fraction: float = 0.2,
    frame_stride: int = 1,
    seed: int = 467,
    formats: tuple[str, ...] = ("yolo", "coco"),
    replace: bool = False,
    exclude_sequences: frozenset[str] = EXCLUDED_SEQUENCES,
) -> dict[str, str]:
    """Build the YOLO and/or COCO versions of UAVDT under dst_dir.

    Args:
        src_dir: The extracted Dataset Ninja download (holds meta.json, train/, test/).
        dst_dir: Folder to write uavdt_yolo/ and uavdt_coco/ into.
        val_fraction: Share of train sequences moved to the validation split.
        frame_stride: Keep every Nth frame of each sequence. Neighbouring video
            frames are nearly identical, so 2-5 cuts size and epoch time at little cost.
        seed: Seed for picking validation sequences, so the split is repeatable.
        formats: Any of "yolo" and "coco".
        replace: Rebuild an output that already exists. Otherwise it is skipped.

    Returns:
        The output folder for each format that was requested.
    """
    assert os.path.isfile(os.path.join(src_dir, "meta.json")), f"EE: no meta.json in {src_dir}"
    assert os.path.isdir(dst_dir), f"EE: destination dir does not exist: {dst_dir}"
    assert 0 < val_fraction < 1, f"EE: val_fraction must be between 0 and 1, got {val_fraction}"
    assert frame_stride >= 1, f"EE: frame_stride must be at least 1, got {frame_stride}"
    unknown = set(formats) - {"yolo", "coco"}
    assert not unknown, f"EE: unknown formats: {sorted(unknown)}"

    out_dirs = {fmt: os.path.join(dst_dir, f"uavdt_{fmt}") for fmt in formats}
    todo = [fmt for fmt in formats if replace or not os.path.isdir(out_dirs[fmt])]
    for fmt in formats:
        if fmt not in todo:
            print(f"Already prepared, skipping (pass replace=True to rebuild): {out_dirs[fmt]}")
    if not todo:
        return out_dirs

    class_map = _class_map(src_dir)
    splits = _split_frames(src_dir, val_fraction, frame_stride, seed, exclude_sequences)
    if exclude_sequences:
        print(f"Excluding {len(exclude_sequences)} bad sequences: {sorted(exclude_sequences)}")
    for split, frames in splits.items():
        print(f"{split}: {len(frames)} images from {len({_sequence(f) for f in frames})} sequences")

    for fmt in todo:
        if os.path.isdir(out_dirs[fmt]):
            shutil.rmtree(out_dirs[fmt])
        print(f"Writing {fmt.upper()} dataset to: {out_dirs[fmt]}")

    writers = []
    if "yolo" in todo:
        writers.append(_YoloWriter(out_dirs["yolo"]))
    if "coco" in todo:
        writers.append(_CocoWriter(out_dirs["coco"]))

    for split, frames in splits.items():
        source = SOURCE_SPLIT[split]
        box_counts = Counter()
        for writer in writers:
            writer.start_split(split)
        for image_name in frames:
            image_path = os.path.join(src_dir, source, "img", image_name)
            record = _read_annotation(os.path.join(src_dir, source, "ann", image_name + ".json"), class_map)
            box_counts.update(CLASSES[box[0]] for box in record["boxes"])
            for writer in writers:
                writer.add_image(split, image_name, image_path, record)
        for writer in writers:
            writer.finish_split(split)
        print(f"  {split} boxes: {dict(box_counts)}")

    for writer in writers:
        writer.finish()
    return out_dirs


def _class_map(src_dir: str) -> dict[int, int]:
    """Map Supervisely class ids to output class indices. Classes not in CLASSES are dropped."""
    with open(os.path.join(src_dir, "meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    by_title = {c["title"]: c["id"] for c in meta["classes"]}
    missing = [name for name in CLASSES if name not in by_title]
    assert not missing, f"EE: classes missing from meta.json: {missing}"
    return {by_title[name]: index for index, name in enumerate(CLASSES)}


def _sequence(image_name: str) -> str:
    # M0203_img000001.jpg -> M0203
    return image_name.split("_", 1)[0]


def _split_frames(
    src_dir: str,
    val_fraction: float,
    frame_stride: int,
    seed: int,
    exclude_sequences: frozenset[str] = EXCLUDED_SEQUENCES,
) -> dict[str, list[str]]:
    """Return the image file names for train, val, and test, in sequence then frame order."""
    by_source = {}
    for source in sorted(set(SOURCE_SPLIT.values())):
        sequences = defaultdict(list)
        for name in sorted(os.listdir(os.path.join(src_dir, source, "img"))):
            if not name.startswith(DETECTION_SEQUENCE_PREFIX):
                continue
            seq = _sequence(name)
            if seq in exclude_sequences:
                continue
            sequences[seq].append(name)
        by_source[source] = {seq: frames[::frame_stride] for seq, frames in sequences.items()}

    train_sequences = sorted(by_source["train"])
    val_count = max(1, round(len(train_sequences) * val_fraction))
    val_sequences = set(random.Random(seed).sample(train_sequences, val_count))

    def frames_of(source: str, keep) -> list[str]:
        return [f for seq in sorted(by_source[source]) if keep(seq) for f in by_source[source][seq]]

    return {
        "train": frames_of("train", lambda seq: seq not in val_sequences),
        "val": frames_of("train", lambda seq: seq in val_sequences),
        "test": frames_of("test", lambda seq: True),
    }


def _read_annotation(path: str, class_map: dict[int, int]) -> dict:
    """Read one Supervisely annotation.

    Returns the image size, its condition tags (night, fog, low alt, ...), and
    its boxes as (class_index, x1, y1, x2, y2) clipped to the image.
    """
    with open(path, encoding="utf-8") as handle:
        ann = json.load(handle)
    width, height = ann["size"]["width"], ann["size"]["height"]

    boxes = []
    for obj in ann["objects"]:
        class_index = class_map.get(obj["classId"])
        if class_index is None or obj["geometryType"] != "rectangle":
            continue
        (ax, ay), (bx, by) = obj["points"]["exterior"]
        x1, x2 = sorted((min(max(ax, 0), width), min(max(bx, 0), width)))
        y1, y2 = sorted((min(max(ay, 0), height), min(max(by, 0), height)))
        if x2 > x1 and y2 > y1:
            boxes.append((class_index, x1, y1, x2, y2))

    # "sequence" carries a value; the condition tags are just present or absent.
    tags = sorted(tag["name"] for tag in ann.get("tags", []) if tag["name"] != "sequence")
    return {"width": width, "height": height, "tags": tags, "boxes": boxes}


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class _YoloWriter:
    def __init__(self, root: str):
        self.root = root

    def start_split(self, split: str) -> None:
        os.makedirs(os.path.join(self.root, "images", split))
        os.makedirs(os.path.join(self.root, "labels", split))

    def add_image(self, split: str, image_name: str, image_path: str, record: dict) -> None:
        _link_or_copy(image_path, os.path.join(self.root, "images", split, image_name))
        width, height = record["width"], record["height"]
        lines = []
        for class_index, x1, y1, x2, y2 in record["boxes"]:
            cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
            w, h = (x2 - x1) / width, (y2 - y1) / height
            lines.append(f"{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        # An empty label file marks a background image with no vehicles.
        label_path = os.path.join(self.root, "labels", split, os.path.splitext(image_name)[0] + ".txt")
        with open(label_path, "w", encoding="utf-8") as handle:
            handle.writelines(lines)

    def finish_split(self, split: str) -> None:
        pass

    def finish(self) -> None:
        # No "path" key: Ultralytics then resolves the splits relative to this
        # file, so the folder still works after being moved or uploaded to Colab.
        lines = [
            "train: images/train\n",
            "val: images/val\n",
            "test: images/test\n",
            f"nc: {len(CLASSES)}\n",
            "names:\n",
            *[f"  {index}: {name}\n" for index, name in enumerate(CLASSES)],
        ]
        with open(os.path.join(self.root, "data.yaml"), "w", encoding="utf-8") as handle:
            handle.writelines(lines)


class _CocoWriter:
    def __init__(self, root: str):
        self.root = root
        self.categories = [
            {"id": index + 1, "name": name, "supercategory": "vehicle"} for index, name in enumerate(CLASSES)
        ]

    def start_split(self, split: str) -> None:
        os.makedirs(os.path.join(self.root, COCO_SPLIT_DIR[split]))
        self.images, self.annotations = [], []

    def add_image(self, split: str, image_name: str, image_path: str, record: dict) -> None:
        _link_or_copy(image_path, os.path.join(self.root, COCO_SPLIT_DIR[split], image_name))
        image_id = len(self.images) + 1
        self.images.append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": record["width"],
                "height": record["height"],
                # Extra fields are ignored by COCO tools; kept for per-condition analysis.
                "sequence": _sequence(image_name),
                "conditions": record["tags"],
            }
        )
        for class_index, x1, y1, x2, y2 in record["boxes"]:
            w, h = x2 - x1, y2 - y1
            self.annotations.append(
                {
                    "id": len(self.annotations) + 1,
                    "image_id": image_id,
                    "category_id": class_index + 1,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )

    def finish_split(self, split: str) -> None:
        payload = {
            "info": {"description": f"UAVDT detection benchmark, {split} split"},
            "categories": self.categories,
            "images": self.images,
            "annotations": self.annotations,
        }
        path = os.path.join(self.root, COCO_SPLIT_DIR[split], "_annotations.coco.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def finish(self) -> None:
        pass


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    prepareUAVDT(os.path.join(data_dir, "uavdt"), data_dir)
