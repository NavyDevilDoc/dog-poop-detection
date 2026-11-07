# Legacy scripts

These scripts built the dataset that the live training pipeline consumes. They
are **not maintained** and **will not run as-is** against the current codebase
(they predate the `config.py` refactor from `ProjectConfig.get("dot.path")` to
typed dataclasses).

They are kept in the repository as a historical record of how the dataset was
created and to document the one-time data-preparation pipeline.

## Why they are retired

The underlying dataset is frozen. All ~1,200 images were annotated by hand with
`annotation_tool.py`, converted to YOLO format, split into train/val, and (once)
augmented. None of that work needs to be repeated. The live code under the
project root (`model_training.py`, `model_testing.py`, `config.py`) is what the
project uses going forward.

## What each script does

| Script | Purpose |
| --- | --- |
| `image_extraction.py` | Pull frames from videos in `raw_media/` and save them as timestamped JPGs. |
| `annotation_tool.py` | Matplotlib-based GUI for drawing bounding boxes on images and saving labelme-style JSON labels. |
| `yolo_conversion.py` | Convert the JSON labels to YOLO `.txt` format and generate `data.yaml`. |
| `dataset_splitting.py` | Split the annotated images into `train/` and `val/` folders in YOLO's expected layout, preserving the val set across retrains. |
| `data_augmentation.py` | Albumentations-based offline augmentation (lighting, geometric, noise). Never ended up being used in the final training run — YOLO's online augmentation did the work. |
| `create_project.py` | Interactive wizard for scaffolding a new multi-class detection project. One-off bootstrapping tool. |
| `combine_images.py` | Small utility to vertically stack two JPGs. Ad-hoc helper. |
| `docs/` | Guides for the annotation tool, the extraction script, and the old project-configuration API. |

## What would need to change to make them run again

- `from config import config` → `from config import load_config`; then
  `cfg = load_config()` and read via attribute access (`cfg.dataset.classes`
  instead of `config.get("dataset.classes")`).
- `from config import get_project_title` → `from config import project_title`;
  the function now takes a `cfg` argument instead of reading a global.
- `config.get_dataset_path("images")` → `cfg.dataset_path("images")`.
- `config.generate_dynamic_colors(...)` → the module-level
  `generate_class_colors(...)` function is still available; the per-instance
  helper is gone.
- `create_project.py` additionally imports `ProjectConfig` and
  `create_project_template()` directly; both were removed. It would need to
  construct a `ProjectConfig` dataclass and call `save_config()`.

If you want to resurrect any of these, grep for its import statements, update
them, and expect a round of fiddling with print statements since the old API's
auto-save-on-set behavior is gone (writes are now explicit via `save_config`).
