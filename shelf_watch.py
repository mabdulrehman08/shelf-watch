"""Count visible COCO products in inferred shelf rows using pretrained YOLOv8."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from statistics import median
import sys

# Keep Ultralytics' writable settings beside this demo, rather than in AppData.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).parent / ".yolo"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mpl"))

PRODUCT_CLASSES = (
    "bottle", "wine glass", "cup", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
)
COLORS = [(53, 211, 255), (255, 183, 77), (121, 220, 120), (210, 140, 255)]


def cluster_rows(items: list[dict], tolerance: float) -> list[list[dict]]:
    """Top-down 1-D clusters; each row's full y-span is <= tolerance pixels.

    Bounding-box bottoms approximate shelf contact points. Limiting the full
    span prevents a chain of nearby detections from merging distant shelves.
    """
    rows: list[list[dict]] = []
    for item in sorted(items, key=lambda d: (d["anchor_y"], d["bbox"][0])):
        if not rows or item["anchor_y"] - rows[-1][0]["anchor_y"] > tolerance:
            rows.append([])
        rows[-1].append(item)
    return rows


def annotate(image, rows: list[list[dict]], planogram: dict):
    import cv2
    import numpy as np

    h, w = image.shape[:2]
    panel_width = 380
    panel_height = max(180, 130 + sum(75 + 27 * len(v) for v in planogram.values()))
    canvas = np.full((max(h, panel_height), w + panel_width, 3), 24, dtype=np.uint8)
    canvas[:h, :w] = image

    def text(label, x, y, color=(240, 240, 240), scale=0.65):
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2, cv2.LINE_AA)

    text("SHELF WATCH / YOLOv8", w + 18, 38, scale=0.75)
    text("Visible detections, inferred rows", w + 18, 70, scale=0.55)
    text(f"{sum(map(len, rows))} items | {len(rows)} rows", w + 18, 100)
    panel_y = 143
    for row_id, row in enumerate(rows, start=1):
        color = COLORS[(row_id - 1) % len(COLORS)]
        anchor = round(median(d["anchor_y"] for d in row))
        cv2.line(canvas, (0, anchor), (w - 1, anchor), color, 2, cv2.LINE_AA)
        text(f"R{row_id}", 8, max(22, anchor - 7), color)
        for item in row:
            x1, y1, x2, y2 = map(round, item["bbox"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"R{row_id} {item['class']} {item['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            lx = max(0, min(x1, w - tw - 5))
            ly = max(th + 5, y1)
            cv2.rectangle(canvas, (lx, ly - th - 5), (lx + tw + 4, ly + 2), (24, 24, 24), -1)
            cv2.putText(canvas, label, (lx + 2, ly - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 1, cv2.LINE_AA)
        text(f"ROW {row_id} | {len(row)} detected", w + 18, panel_y, color)
        panel_y += 31
        for name, count in planogram[f"row_{row_id}"].items():
            text(f"{name}: {count}", w + 28, panel_y)
            panel_y += 27
        panel_y += 44
    if not rows:
        text("No matching products detected.", w + 18, panel_y, scale=0.55)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="One or more local shelf/fridge photos")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights (downloaded on first run)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--row-tolerance", type=float, default=0.08,
                        help="Maximum row y-span as fraction of image height (default: 0.08)")
    parser.add_argument("--anchor", choices=("bottom", "center"), default="bottom")
    parser.add_argument("--device", default="cpu", help="cpu, 0 for first CUDA GPU, etc.")
    parser.add_argument("--classes", nargs="+", default=list(PRODUCT_CLASSES), help="COCO names to count")
    args = parser.parse_args()
    if not 0 < args.conf <= 1 or not 0 < args.row_tolerance <= 1 or args.imgsz <= 0:
        parser.error("conf and row-tolerance must be in (0, 1]; imgsz must be positive")
    if len({p.stem.casefold() for p in args.images}) != len(args.images):
        parser.error("Input filenames must have distinct stems to avoid output collisions")
    for path in args.images:
        if not path.is_file():
            parser.error(f"Image does not exist: {path}")

    import cv2
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(args.model)
    class_ids = [i for i, name in model.names.items() if name in args.classes]
    unknown = set(args.classes) - set(model.names.values())
    if unknown:
        parser.error(f"Classes absent from model: {', '.join(sorted(unknown))}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            parser.error(f"Cannot decode image: {path}")
        result = model.predict(image, conf=args.conf, imgsz=args.imgsz,
                               classes=class_ids, device=args.device, verbose=False)[0]
        items = []
        for box in result.boxes:
            bbox = box.xyxy[0].cpu().tolist()
            items.append({"class": model.names[int(box.cls.item())],
                          "confidence": round(float(box.conf.item()), 4),
                          "bbox": [round(v, 2) for v in bbox],
                          "anchor_y": bbox[3] if args.anchor == "bottom" else (bbox[1] + bbox[3]) / 2})
        rows = cluster_rows(items, args.row_tolerance * image.shape[0])
        planogram = {f"row_{i}": dict(sorted(Counter(d["class"] for d in row).items()))
                     for i, row in enumerate(rows, start=1)}
        prefix = args.output_dir / path.stem
        annotated = prefix.with_name(prefix.name + "_annotated.jpg")
        if not cv2.imwrite(str(annotated), annotate(image, rows, planogram)):
            raise OSError(f"Cannot write {annotated}")
        prefix.with_name(prefix.name + ".json").write_text(json.dumps(planogram, indent=2) + "\n", encoding="utf-8")
        details = {"image": path.as_posix(), "image_size": {"width": image.shape[1], "height": image.shape[0]},
                   "model": str(args.model), "ultralytics_version": ultralytics.__version__,
                   "conf": args.conf, "imgsz": args.imgsz, "device": args.device,
                   "anchor": args.anchor, "row_tolerance": args.row_tolerance,
                   "classes": args.classes, "total_detections": len(items),
                   "rows": {f"row_{i}": row for i, row in enumerate(rows, start=1)}}
        prefix.with_name(prefix.name + "_detections.json").write_text(
            json.dumps(details, indent=2) + "\n", encoding="utf-8")
        print(f"{path.name}: {len(items)} detections, {len(rows)} inferred rows -> {annotated}")
        print(json.dumps(planogram))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"shelf_watch: {exc}", file=sys.stderr)
        sys.exit(1)
