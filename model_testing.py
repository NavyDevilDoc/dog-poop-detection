"""Run a trained YOLO detector on images.

Usage (CLI):
    python model_testing.py <image...>                     # batch detect, conf 0.5
    python model_testing.py <image...> --conf 0.3          # custom confidence
    python model_testing.py <image...> --model path/to/best.pt
    python model_testing.py <image...> --multi-scale       # run at 3 scales, dedupe
    python model_testing.py <image...> --sweep             # threshold sweep
    python model_testing.py <image...> --save-dir out/     # save annotated images
    python model_testing.py <image...> --gallery           # show matplotlib gallery

If --model is omitted, the newest best.pt under runs/detect/train/*/weights/
is used automatically.

Importable API:
    from model_testing import load_model, detect, batch_detect
    model = load_model("runs/detect/train/poop_detector2_v2/weights/best.pt")
    detections = detect(model, "image.jpg", confidence=0.4)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


Bbox = tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass
class Detection:
    confidence: float
    class_id: int
    class_name: str
    bbox: Bbox


@dataclass
class ImageResult:
    image_path: Path
    detections: list[Detection]

    @property
    def found(self) -> bool:
        return bool(self.detections)

    @property
    def max_confidence(self) -> float:
        return max((d.confidence for d in self.detections), default=0.0)


@dataclass
class BatchResult:
    results: list[ImageResult] = field(default_factory=list)

    @property
    def total_images(self) -> int:
        return len(self.results)

    @property
    def images_with_detections(self) -> int:
        return sum(1 for r in self.results if r.found)

    @property
    def total_detections(self) -> int:
        return sum(len(r.detections) for r in self.results)


@dataclass
class ThresholdResult:
    threshold: float
    images_with_detections: int
    total_detections: int
    detection_rate: float


def calculate_iou(box_a: Bbox, box_b: Bbox) -> float:
    """Intersection-over-union for two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def remove_duplicate_detections(
    detections: list[Detection], iou_threshold: float = 0.5
) -> list[Detection]:
    """Keep highest-confidence detection per cluster of overlapping boxes."""
    if len(detections) <= 1:
        return list(detections)
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for det in sorted_dets:
        if not any(calculate_iou(det.bbox, existing.bbox) > iou_threshold for existing in kept):
            kept.append(det)
    return kept


def find_latest_model(search_dir: Path | str = "runs/detect/train") -> Path | None:
    """Return the most recently modified ``*/weights/best.pt`` under search_dir, or None."""
    search_dir = Path(search_dir)
    if not search_dir.exists():
        return None
    candidates = list(search_dir.glob("*/weights/best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model(model_path: Path | str) -> YOLO:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path))


def _detections_from_result(result, class_names: dict[int, str]) -> list[Detection]:
    detections: list[Detection] = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        detections.append(
            Detection(
                confidence=float(box.conf[0].item()),
                class_id=cls_id,
                class_name=class_names.get(cls_id, str(cls_id)),
                bbox=tuple(box.xyxy[0].cpu().numpy().tolist()),  # type: ignore[arg-type]
            )
        )
    return detections


def detect(
    model: YOLO,
    image_path: Path | str,
    *,
    confidence: float = 0.5,
    img_size: int = 640,
) -> list[Detection]:
    """Run the model on one image and return detections above the confidence threshold."""
    results = model(str(image_path), conf=confidence, imgsz=img_size, verbose=False)
    return _detections_from_result(results[0], getattr(model, "names", {}))


def detect_multi_scale(
    model: YOLO,
    image_path: Path | str,
    *,
    confidence: float = 0.4,
    scales: list[int] | None = None,
    iou_dedupe: float = 0.5,
) -> list[Detection]:
    """Run the model at multiple scales and merge results, deduping by IoU."""
    scales = scales or [416, 640, 832]
    all_detections: list[Detection] = []
    for scale in scales:
        all_detections.extend(detect(model, image_path, confidence=confidence, img_size=scale))
    return remove_duplicate_detections(all_detections, iou_threshold=iou_dedupe)


def batch_detect(
    model: YOLO,
    image_paths: list[Path | str],
    *,
    confidence: float = 0.5,
    img_size: int = 640,
    multi_scale: bool = False,
) -> BatchResult:
    """Run detection on each image and collect results."""
    results = BatchResult()
    for path in image_paths:
        path = Path(path)
        if multi_scale:
            dets = detect_multi_scale(model, path, confidence=confidence)
        else:
            dets = detect(model, path, confidence=confidence, img_size=img_size)
        results.results.append(ImageResult(image_path=path, detections=dets))
    return results


def sweep_confidence_thresholds(
    model: YOLO,
    image_paths: list[Path | str],
    thresholds: list[float] | None = None,
) -> list[ThresholdResult]:
    """Run detection at several confidence thresholds and summarize detection counts."""
    thresholds = thresholds or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    n = len(image_paths)
    sweep: list[ThresholdResult] = []
    for thresh in thresholds:
        result = batch_detect(model, image_paths, confidence=thresh)
        sweep.append(
            ThresholdResult(
                threshold=thresh,
                images_with_detections=result.images_with_detections,
                total_detections=result.total_detections,
                detection_rate=result.images_with_detections / n if n else 0.0,
            )
        )
    return sweep


def save_annotated_image(
    model: YOLO,
    image_path: Path | str,
    output_dir: Path | str,
    *,
    confidence: float = 0.5,
    img_size: int = 640,
) -> Path:
    """Run the model and save the plotted-detections image to output_dir."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = model(str(image_path), conf=confidence, imgsz=img_size, verbose=False)
    annotated = results[0].plot()
    out_path = output_dir / f"detected_{image_path.name}"
    cv2.imwrite(str(out_path), annotated)
    return out_path


def print_summary(batch: BatchResult) -> None:
    print(f"Images tested:         {batch.total_images}")
    print(f"Images with detection: {batch.images_with_detections}")
    print(f"Images clean:          {batch.total_images - batch.images_with_detections}")
    print(f"Total detections:      {batch.total_detections}")
    if batch.total_images:
        rate = batch.images_with_detections / batch.total_images
        print(f"Detection rate:        {rate:.1%}")
    print()
    for i, result in enumerate(batch.results, 1):
        status = "DETECTED" if result.found else "clean"
        suffix = f" (max conf {result.max_confidence:.3f})" if result.found else ""
        print(f"  {i:2d}. {result.image_path.name}: {status}{suffix}")


def print_sweep(sweep: list[ThresholdResult], n_images: int) -> None:
    print(f"Confidence sweep over {n_images} images:")
    print(f"  {'thresh':>6}  {'det_rate':>8}  {'img_hit':>7}  {'total_det':>9}")
    for r in sweep:
        print(f"  {r.threshold:>6.2f}  {r.detection_rate:>7.1%}   {r.images_with_detections:>7d}  {r.total_detections:>9d}")


def show_gallery(batch: BatchResult, max_images: int = 9) -> None:
    """Display results in a matplotlib grid (blocking)."""
    results = batch.results[:max_images]
    n = len(results)
    if n == 0:
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), squeeze=False)
    flat_axes = axes.flatten()

    for i, result in enumerate(results):
        img = cv2.imread(str(result.image_path))
        if img is None:
            flat_axes[i].text(0.5, 0.5, f"Could not load\n{result.image_path.name}",
                              ha="center", va="center")
            flat_axes[i].axis("off")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        flat_axes[i].imshow(img_rgb)
        status = "DETECTED" if result.found else "clean"
        suffix = f"\nmax conf {result.max_confidence:.3f}" if result.found else ""
        flat_axes[i].set_title(f"{result.image_path.name}\n{status}{suffix}", fontsize=10)
        flat_axes[i].axis("off")
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor="red", facecolor="none")
            flat_axes[i].add_patch(rect)

    for i in range(n, len(flat_axes)):
        flat_axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a trained YOLO detector on images.",
    )
    parser.add_argument("images", nargs="*", help="One or more image paths to analyze")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a .pt file (defaults to the newest runs/detect/train/*/weights/best.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default 0.5)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default 640)")
    parser.add_argument("--multi-scale", action="store_true",
                        help="Run detection at multiple scales and dedupe overlaps")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep confidence thresholds 0.1-0.7 and report detection counts")
    parser.add_argument("--save-dir", default=None, help="If set, save annotated images here")
    parser.add_argument("--gallery", action="store_true", help="Show a matplotlib results gallery")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if not args.images:
        parser.error("at least one image path is required")

    model_path = Path(args.model) if args.model else find_latest_model()
    if model_path is None:
        print("No model found. Pass --model /path/to/best.pt or train one first.", file=sys.stderr)
        return 1
    print(f"Using model: {model_path}")
    model = load_model(model_path)

    if args.sweep:
        sweep = sweep_confidence_thresholds(model, args.images)
        print_sweep(sweep, len(args.images))
        return 0

    batch = batch_detect(
        model,
        args.images,
        confidence=args.conf,
        img_size=args.imgsz,
        multi_scale=args.multi_scale,
    )
    print_summary(batch)

    if args.save_dir:
        for result in batch.results:
            out = save_annotated_image(
                model, result.image_path, args.save_dir,
                confidence=args.conf, img_size=args.imgsz,
            )
            print(f"  saved {out}")

    if args.gallery:
        show_gallery(batch)

    return 0


if __name__ == "__main__":
    sys.exit(main())
