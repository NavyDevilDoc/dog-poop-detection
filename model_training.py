"""Train a YOLO detector on the poop detection dataset.

Usage (CLI):
    python model_training.py --epochs 50               # train with defaults
    python model_training.py --model s --epochs 100    # yolo11s, 100 epochs
    python model_training.py --model runs/detect/train/poop_detector2_v2/weights/best.pt
                                                       # continue from existing weights

Importable API:
    from model_training import train, next_run_name, validate_dataset_config
    train(data_yaml=..., run_name=..., epochs=...)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from config import load_config


MODEL_VARIANTS: dict[str, str] = {
    "n": "yolo11n.pt",
    "s": "yolo11s.pt",
    "m": "yolo11m.pt",
    "l": "yolo11l.pt",
    "x": "yolo11x.pt",
}


def validate_dataset_config(data_yaml: Path | str) -> bool:
    """Return True if data.yaml is well-formed and the referenced dirs exist."""
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        print(f"Config file not found: {data_yaml}")
        return False

    with data_yaml.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    required = ["path", "train", "val", "nc", "names"]
    missing = [k for k in required if k not in config]
    if missing:
        print(f"Missing required keys in {data_yaml}: {missing}")
        return False

    base = Path(config["path"])
    train_dir = base / config["train"]
    val_dir = base / config["val"]
    if not train_dir.exists():
        print(f"Training directory not found: {train_dir}")
        return False
    if not val_dir.exists():
        print(f"Validation directory not found: {val_dir}")
        return False

    train_count = len(list(train_dir.glob("*.jpg")))
    val_count = len(list(val_dir.glob("*.jpg")))
    print(f"Dataset: {train_count} train / {val_count} val, {config['nc']} classes {config['names']}")
    return True


def next_run_name(project_dir: Path | str, base_name: str = "poop_detector") -> str:
    """Return the next available run-directory name under project_dir.

    Scans subdirectories of project_dir for names matching ``base_name`` or
    ``base_name_v<N>``. Returns ``base_name_v<N+1>`` where N is the highest
    existing version, or ``base_name_v1`` if nothing matching exists.
    """
    project_dir = Path(project_dir)
    if not project_dir.exists():
        return f"{base_name}_v1"

    prefix = f"{base_name}_v"
    max_version = 0
    for child in project_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith(prefix):
            try:
                version = int(child.name[len(prefix):])
            except ValueError:
                continue
            max_version = max(max_version, version)
        elif child.name == base_name:
            max_version = max(max_version, 1)
    return f"{base_name}_v{max_version + 1}" if max_version else f"{base_name}_v1"


def train(
    *,
    data_yaml: Path | str,
    run_name: str,
    model: str = "n",
    epochs: int = 50,
    img_size: int = 640,
    batch_size: int = 16,
    project_dir: Path | str = "runs/detect/train",
    save_period: int = 10,
) -> Path:
    """Train a YOLO detector. Returns the path to best.pt.

    ``model`` may be a size key in MODEL_VARIANTS ('n'|'s'|'m'|'l'|'x') or a
    path to an existing .pt file to continue training from.
    """
    model_ref = MODEL_VARIANTS.get(model, model)
    print(f"Training: model={model_ref} epochs={epochs} imgsz={img_size} batch={batch_size}")
    print(f"Run: {project_dir}/{run_name}")

    yolo = YOLO(model_ref)
    yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name=run_name,
        project=str(project_dir),
        save=True,
        save_period=save_period,
        plots=True,
        verbose=True,
    )
    best = Path(project_dir) / run_name / "weights" / "best.pt"
    print(f"Best model saved to: {best}")
    return best


def validate_trained_model(model_path: Path | str, data_yaml: Path | str) -> dict[str, float]:
    """Run validation metrics on a trained model and return a plain dict."""
    print(f"Validating {model_path}")
    yolo = YOLO(str(model_path))
    metrics = yolo.val(data=str(data_yaml))
    result = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }
    for k, v in result.items():
        print(f"  {k}: {v:.3f}")
    return result


def run_inference_on_image(
    model_path: Path | str,
    image_path: Path | str,
    output_dir: Path | str | None = None,
) -> Path:
    """Run a trained model on one image; save the plotted result and return its path."""
    image_path = Path(image_path)
    print(f"Inference on {image_path}")
    yolo = YOLO(str(model_path))
    results = yolo(str(image_path))

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            print(f"  {len(boxes)} detections")
            for i, box in enumerate(boxes):
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                print(f"    {i + 1}: class={int(cls)} conf={conf:.3f}")
        else:
            print("  no detections")

    out_dir = Path(output_dir) if output_dir else image_path.parent
    out_path = out_dir / f"detected_{image_path.name}"
    cv2.imwrite(str(out_path), results[0].plot())
    print(f"  saved {out_path}")
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a YOLO detector on the poop detection dataset.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to data.yaml (defaults to <dataset.base>/data.yaml from project_config.yaml)",
    )
    parser.add_argument(
        "--model",
        default="n",
        help="Size key 'n|s|m|l|x' (yolo11 variants) or path to a .pt file to continue from",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/detect/train", help="Base directory for training runs")
    parser.add_argument("--name", default=None, help="Run name (auto-generated if omitted)")
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument(
        "--post-check",
        action="store_true",
        default=True,
        help="After training, run validation and sample inference (default: on)",
    )
    parser.add_argument("--no-post-check", dest="post_check", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()

    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if args.data is None:
        cfg = load_config()
        data_yaml = cfg.dataset_path() / "data.yaml"
    else:
        data_yaml = Path(args.data)

    if not validate_dataset_config(data_yaml):
        return 1

    run_name = args.name or next_run_name(args.project)

    best = train(
        data_yaml=data_yaml,
        run_name=run_name,
        model=args.model,
        epochs=args.epochs,
        img_size=args.imgsz,
        batch_size=args.batch,
        project_dir=args.project,
        save_period=args.save_period,
    )

    if args.post_check:
        validate_trained_model(best, data_yaml)
        val_images = list((Path(data_yaml).parent / "val" / "images").glob("*.jpg"))
        if val_images:
            run_inference_on_image(best, val_images[0])

    print(f"\nDone. Best model: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
