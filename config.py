"""Typed project configuration loaded from project_config.yaml.

The config is a tree of dataclasses. Callers load it with ``load_config()``,
read fields via attribute access (``cfg.dataset.classes``), and persist
changes with ``save_config(cfg)``. There is no module-level singleton
and no implicit disk I/O on attribute writes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Type, TypeVar

import matplotlib.colors as mcolors
import yaml


_COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def generate_class_colors(classes: list[str]) -> dict[str, str]:
    """Return a mapping from class name to a visually distinct hex color."""
    palette = list(_COLOR_PALETTE)
    if len(classes) > len(palette):
        extras = [
            c for c in mcolors.CSS4_COLORS.values()
            if not any(light in c.lower() for light in ("white", "light", "pale", "ivory", "snow"))
        ]
        palette.extend(extras)
    return {cls: palette[i % len(palette)] for i, cls in enumerate(classes)}


@dataclass
class ProjectMeta:
    name: str = "poop_detection"
    version: str = "1.0.0"
    description: str = "Multi-class dog poop detection system"
    created: str = ""


@dataclass
class DatasetConfig:
    name: str = "poop_detection_dataset"
    classes: list[str] = field(default_factory=lambda: ["leaf", "poop", "stick"])
    class_colors: dict[str, str] = field(default_factory=lambda: {
        "leaf": "green",
        "poop": "red",
        "stick": "brown",
    })


@dataclass
class DirectoryConfig:
    base: str = "poop_detection_dataset"
    images: str = "images"
    labels: str = "labels"
    yolo_labels: str = "yolo_labels"
    train: str = "train"
    val: str = "val"
    test: str = "test"


@dataclass
class ExtractionConfig:
    target_frames_per_video: int = 15
    output_format: str = "jpg"
    quality: int = 95
    preview_samples: int = 6


@dataclass
class AnnotationConfig:
    default_class: str = "poop"
    auto_save: bool = True
    validation_on_save: bool = True


@dataclass
class YoloConfig:
    format_version: str = "8"
    coordinate_format: str = "normalized"
    min_annotation_rate: float = 0.2


@dataclass
class SplittingConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    random_seed: int = 42
    stratified: bool = False


@dataclass
class PatternsConfig:
    image_extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png"])
    video_extensions: list[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])
    timestamp_format: str = "%Y%m%d_%H%M%S"


T = TypeVar("T")


def _from_dict(cls: Type[T], data: dict | None) -> T:
    """Build a dataclass from a dict, silently dropping unknown keys."""
    if not data:
        return cls()
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ProjectConfig:
    project: ProjectMeta = field(default_factory=ProjectMeta)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    yolo: YoloConfig = field(default_factory=YoloConfig)
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    patterns: PatternsConfig = field(default_factory=PatternsConfig)

    def dataset_path(self, subdir: str | None = None) -> Path:
        """Return the dataset base path, optionally joined with a known subdir alias."""
        base = Path(self.directories.base)
        if subdir is None:
            return base
        resolved = {
            "images": self.directories.images,
            "labels": self.directories.labels,
            "yolo_labels": self.directories.yolo_labels,
            "train": self.directories.train,
            "val": self.directories.val,
            "test": self.directories.test,
        }.get(subdir, subdir)
        return base / resolved


DEFAULT_CONFIG_PATH = Path("project_config.yaml")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Load a ProjectConfig from a YAML file, or return a default if the file is absent."""
    path = Path(path)
    if not path.exists():
        return ProjectConfig(project=ProjectMeta(created=datetime.now().isoformat()))

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return ProjectConfig(
        project=_from_dict(ProjectMeta, raw.get("project")),
        dataset=_from_dict(DatasetConfig, raw.get("dataset")),
        directories=_from_dict(DirectoryConfig, raw.get("directories")),
        extraction=_from_dict(ExtractionConfig, raw.get("extraction")),
        annotation=_from_dict(AnnotationConfig, raw.get("annotation")),
        yolo=_from_dict(YoloConfig, raw.get("yolo")),
        splitting=_from_dict(SplittingConfig, raw.get("splitting")),
        patterns=_from_dict(PatternsConfig, raw.get("patterns")),
    )


def save_config(config: ProjectConfig, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
    """Serialize a ProjectConfig to YAML."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=True, indent=2)


def project_title(config: ProjectConfig, context: str = "general") -> str:
    """Context-specific project title string (no emojis, cross-platform safe)."""
    description = config.project.description.title()
    contexts = {
        "extraction": f"{description} - Frame Extraction",
        "annotation": f"{description} - Annotation Tool",
        "training": f"{description} - Model Training",
        "testing": f"{description} - Model Testing",
    }
    return contexts.get(context, description)
