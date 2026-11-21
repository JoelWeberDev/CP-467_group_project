"""
@Description: Opens annotated UAVDT images so the boxes can be checked by eye.

Keys:
- n, space, or right arrow: next frame
- p or left arrow: previous frame
- ] : first frame of the next sequence
- [ : first frame of the previous sequence
- q or escape: quit
"""

import json
import os
import sys

# This OpenCV build has no Wayland plugin, and the desktop sets QT_QPA_PLATFORM
# to wayland first. X11 is available through XWayland.
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2


def previewDataset(data_path: str, split: str = "train", start: int = 0):
    dataset_dir = _dataset_dir(data_path)
    assert split in ("train", "test"), f"EE: split must be 'train' or 'test', got {split}"

    colors = _class_colors(os.path.join(dataset_dir, "meta.json"))
    names = _image_names(os.path.join(dataset_dir, split, "img"))
    assert names, f"EE: no images in {os.path.join(dataset_dir, split, 'img')}"
    assert 0 <= start < len(names), f"EE: start {start} is outside 0..{len(names) - 1}"

    print(
        f"II: {len(names)} {split} images. "
        "n/right next, p/left previous, ] next sequence, [ previous sequence, q quit."
    )

    index = start
    window = "UAVDT"
    while True:
        name = names[index]
        image = _annotated_image(dataset_dir, split, name, colors)
        title = f"{split} {index + 1}/{len(names)}  {name}  {_sequence_name(name)}"
        cv2.imshow(window, image)
        cv2.setWindowTitle(window, title)

        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27):
            break
        elif key in (ord("n"), ord(" "), 83, 65363, 2555904):
            index = (index + 1) % len(names)
        elif key in (ord("p"), 81, 65361, 2424832):
            index = (index - 1) % len(names)
        elif key == ord("]"):
            index = _step_sequence(names, index, 1)
        elif key == ord("["):
            index = _step_sequence(names, index, -1)

    cv2.destroyAllWindows()


def _dataset_dir(data_path: str) -> str:
    assert os.path.isdir(data_path), f"EE: data path does not exist: {data_path}"
    if os.path.isfile(os.path.join(data_path, "meta.json")):
        return data_path
    nested = os.path.join(data_path, "uavdt")
    assert os.path.isdir(nested), f"EE: UAVDT dataset not found at {data_path}"
    return nested


def _class_colors(meta_path: str) -> dict[str, tuple[int, int, int]]:
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    colors = {}
    for item in meta.get("classes", []):
        colors[item["title"]] = _hex_to_bgr(item.get("color", "#FFFFFF"))
    return colors


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    red, green, blue = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return blue, green, red


def _image_names(img_dir: str) -> list[str]:
    names = [
        name
        for name in os.listdir(img_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    names.sort()
    return names


def _annotated_image(dataset_dir: str, split: str, name: str, colors: dict[str, tuple[int, int, int]]):
    image_path = os.path.join(dataset_dir, split, "img", name)
    image = cv2.imread(image_path)
    assert image is not None, f"EE: could not read {image_path}"

    annotation_path = os.path.join(dataset_dir, split, "ann", f"{name}.json")
    with open(annotation_path, encoding="utf-8") as handle:
        annotation = json.load(handle)

    _draw_tag_bar(image, annotation.get("tags", []))
    for obj in annotation.get("objects", []):
        if obj.get("geometryType") != "rectangle":
            continue
        exterior = obj.get("points", {}).get("exterior")
        if not exterior or len(exterior) < 2:
            continue
        x1, y1 = exterior[0]
        x2, y2 = exterior[1]
        left, right = sorted((int(x1), int(x2)))
        top, bottom = sorted((int(y1), int(y2)))
        title = obj.get("classTitle", "object")
        color = colors.get(title, (255, 255, 255))
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        _draw_label(image, title, left, top, color)
    return image


def _draw_tag_bar(image, tags: list) -> None:
    labels = []
    for tag in tags:
        name = tag.get("name", "")
        value = tag.get("value")
        labels.append(name if value in (None, "") else f"{name}={value}")
    if not labels:
        return
    text = "  ".join(labels)
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_label(image, text: str, left: int, top: int, color: tuple[int, int, int]) -> None:
    scale = 0.45
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y = top - 4 if top - height - 6 >= 28 else top + height + 4
    cv2.rectangle(image, (left, y - height - 2), (left + width + 4, y + baseline), color, -1)
    cv2.putText(
        image,
        text,
        (left + 2, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def _sequence_name(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    marker = "_img"
    return stem.split(marker)[0] if marker in stem else stem


def _step_sequence(names: list[str], index: int, step: int) -> int:
    current = _sequence_name(names[index])
    cursor = index
    while 0 <= cursor < len(names) and _sequence_name(names[cursor]) == current:
        cursor += step
    if step < 0 and cursor >= 0:
        previous = _sequence_name(names[cursor])
        while cursor > 0 and _sequence_name(names[cursor - 1]) == previous:
            cursor -= 1
    return min(max(cursor, 0), len(names) - 1)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, "data")
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    previewDataset(data_path, split, start)
