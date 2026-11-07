from pathlib import Path

import pytest
import yaml

from config import (
    DatasetConfig,
    ProjectConfig,
    ProjectMeta,
    generate_class_colors,
    load_config,
    project_title,
    save_config,
)


def test_load_returns_defaults_when_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    cfg = load_config(missing)
    assert isinstance(cfg, ProjectConfig)
    assert cfg.dataset.classes == ["leaf", "poop", "stick"]
    assert cfg.dataset.name == "poop_detection_dataset"


def test_load_defaults_populate_created_timestamp(tmp_path):
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg.project.created != ""


def test_load_parses_populated_yaml(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        yaml.dump({
            "project": {"name": "demo", "description": "demo project"},
            "dataset": {"classes": ["cat", "dog"], "name": "demo_dataset"},
            "splitting": {"train_ratio": 0.7, "val_ratio": 0.3},
        })
    )
    cfg = load_config(yaml_path)
    assert cfg.project.name == "demo"
    assert cfg.dataset.classes == ["cat", "dog"]
    assert cfg.splitting.train_ratio == 0.7


def test_save_then_load_is_lossless(tmp_path):
    original = ProjectConfig(
        project=ProjectMeta(name="rt", description="round-trip"),
        dataset=DatasetConfig(name="rt_data", classes=["a", "b"]),
    )
    yaml_path = tmp_path / "rt.yaml"
    save_config(original, yaml_path)
    reloaded = load_config(yaml_path)
    assert reloaded == original


def test_unknown_yaml_keys_are_ignored(tmp_path):
    yaml_path = tmp_path / "with_junk.yaml"
    yaml_path.write_text(
        yaml.dump({
            "dataset": {"classes": ["x"], "name": "d", "legacy_rock_field": "gray"},
            "unknown_top_level_section": {"foo": "bar"},
        })
    )
    cfg = load_config(yaml_path)
    assert cfg.dataset.classes == ["x"]
    assert cfg.dataset.name == "d"


def test_dataset_path_with_known_alias():
    cfg = ProjectConfig()
    assert cfg.dataset_path("images") == Path("poop_detection_dataset") / "images"
    assert cfg.dataset_path("train") == Path("poop_detection_dataset") / "train"


def test_dataset_path_with_unknown_subdir_uses_literal_name():
    cfg = ProjectConfig()
    assert cfg.dataset_path("some_custom_dir") == Path("poop_detection_dataset") / "some_custom_dir"


def test_dataset_path_without_subdir_returns_base():
    cfg = ProjectConfig()
    assert cfg.dataset_path() == Path("poop_detection_dataset")


def test_generate_class_colors_assigns_distinct_colors():
    colors = generate_class_colors(["a", "b", "c"])
    assert set(colors.keys()) == {"a", "b", "c"}
    assert len(set(colors.values())) == 3


def test_generate_class_colors_handles_more_than_base_palette():
    classes = [f"class_{i}" for i in range(20)]
    colors = generate_class_colors(classes)
    assert set(colors.keys()) == set(classes)
    # Every class should have a color assigned
    assert all(colors[c].startswith("#") or colors[c].isalpha() for c in classes)


@pytest.mark.parametrize(
    "context,expected_fragment",
    [
        ("training", "Model Training"),
        ("testing", "Model Testing"),
        ("extraction", "Frame Extraction"),
        ("annotation", "Annotation Tool"),
        ("unknown_context", "Multi-Class Dog Poop Detection System"),
    ],
)
def test_project_title_uses_context(context, expected_fragment):
    cfg = ProjectConfig()
    title = project_title(cfg, context)
    assert expected_fragment in title
