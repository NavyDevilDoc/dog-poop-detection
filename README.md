# Dog Poop Detection

A YOLO object detector trained to find dog waste in yards full of fall leaves —
the time of year when it's hardest to spot by eye.

The project started as a weekend effort to build a phone app for the yard-cleanup
problem. The phone-app side was never finished, but the detection model works
and the code has been rebuilt into a small, typed, scriptable library.

## Status

- **Dataset:** 1,237 hand-annotated images (969 train / 268 val) across 3 classes
  (`leaf`, `poop`, `stick`). Frozen — not in this repo (too large and personal).
- **Trained model:** YOLOv8m, ~0.85 mAP50 at peak after 50 epochs continuing
  from an earlier run. Weights are not in this repo; a published version would
  attach `best.pt` as a release asset.
- **Mobile deployment:** none. Original Android-app goal was shelved after the
  author switched to iPhone.
- **Current focus:** code quality. The source was recently refactored from a
  pile of interactive scripts into a typed library with CLIs and unit tests.

## Repository layout

```
.
├── config.py               # Typed project config (dataclass tree)
├── model_training.py       # train() + CLI; builds/continues a YOLO run
├── model_testing.py        # detect() + CLI; batch inference, sweeps, gallery
├── check_torch_cuda.py     # GPU diagnostic utility
├── project_config.yaml     # Dataset / directory / class registry
├── pyproject.toml          # pytest configuration
├── requirements.txt        # Runtime + dev dependencies
├── tests/                  # 40 unit tests (pytest)
└── legacy/                 # One-time dataset-building scripts (see its README)
```

Running any CLI script (`model_training.py`, `model_testing.py`) with no
arguments prints its help and exits cleanly.

## Quick start

```bash
# Set up the environment
python -m venv venv
# Windows:  venv\Scripts\activate
# Unix:     source venv/bin/activate
pip install -r requirements.txt

# Confirm the test suite passes
pytest

# Run inference on an image (requires a trained weights file)
python model_testing.py photo.jpg --conf 0.4

# Sweep confidence thresholds across a set of images
python model_testing.py photos/*.jpg --sweep

# Train a model (requires a labeled dataset in YOLO format)
python model_training.py --epochs 50 --model n
```

If you have an NVIDIA GPU and want CUDA-accelerated PyTorch, run
`python check_torch_cuda.py` for a diagnosis and install guidance.

## Model performance

Best checkpoint: `runs/detect/train/poop_detector2_v2/weights/best.pt`
(YOLOv8m, 50 epochs, batch 32, imgsz 640, continuation of a prior run).

| Metric    |  Value |
| --------- | -----: |
| Precision |   0.90 |
| Recall    |   0.81 |
| mAP50     |   0.85 |
| mAP50-95  |   0.55 |

With the existing frozen dataset, a reasonable ceiling is probably ~0.90 mAP50
via a larger backbone (`yolo11l` or `yolo11x`), a higher input resolution
(800–1024 px), and a label-quality pass on the hardest validation examples.
Significant improvements beyond that would need more data — especially hard
cases like waste partially hidden under leaves.

## API usage

Both training and testing are importable, not just command-line scripts.

```python
from config import load_config
from model_training import train, next_run_name
from model_testing import load_model, batch_detect, print_summary

cfg = load_config()
run_name = next_run_name("runs/detect/train")
best = train(
    data_yaml=cfg.dataset_path() / "data.yaml",
    run_name=run_name,
    model="n",
    epochs=50,
)

model = load_model(best)
batch = batch_detect(model, ["img1.jpg", "img2.jpg"], confidence=0.4)
print_summary(batch)
```

## Development

```bash
# Full suite
pytest

# One file, verbose
pytest tests/test_config.py -v
```

Written against Python 3.10+. There is no linter or formatter wired up yet.

## Dataset provenance

The dataset was built from scratch using a set of one-off scripts now
preserved under `legacy/`: frame extraction from phone videos, hand annotation
in a custom matplotlib GUI (no automated labeler was available at the time),
conversion to YOLO format, splitting into train/val, and offline augmentation
(which ended up unused — YOLO's online augmentation did the work). See
[legacy/README.md](legacy/README.md) for details.

Those scripts predate the `config.py` dataclass refactor and will not run
against the current API without updates.
