from pathlib import Path

import yaml

from model_training import next_run_name, validate_dataset_config


def test_next_run_name_when_project_dir_missing(tmp_path):
    assert next_run_name(tmp_path / "nope") == "poop_detector_v1"


def test_next_run_name_when_project_dir_empty(tmp_path):
    assert next_run_name(tmp_path) == "poop_detector_v1"


def test_next_run_name_increments_from_highest_version(tmp_path):
    (tmp_path / "poop_detector_v1").mkdir()
    (tmp_path / "poop_detector_v3").mkdir()
    (tmp_path / "poop_detector_v2").mkdir()
    assert next_run_name(tmp_path) == "poop_detector_v4"


def test_next_run_name_ignores_non_version_suffixes(tmp_path):
    (tmp_path / "poop_detector_v2").mkdir()
    (tmp_path / "poop_detector_vFOO").mkdir()
    (tmp_path / "unrelated_dir").mkdir()
    (tmp_path / "poop_detector_v99").mkdir()
    assert next_run_name(tmp_path) == "poop_detector_v100"


def test_next_run_name_ignores_files(tmp_path):
    (tmp_path / "poop_detector_v7").mkdir()
    (tmp_path / "poop_detector_v99.txt").write_text("not a dir")
    assert next_run_name(tmp_path) == "poop_detector_v8"


def test_next_run_name_handles_unversioned_base(tmp_path):
    (tmp_path / "poop_detector").mkdir()
    assert next_run_name(tmp_path) == "poop_detector_v2"


def test_next_run_name_custom_base_name(tmp_path):
    (tmp_path / "experiment_v5").mkdir()
    assert next_run_name(tmp_path, base_name="experiment") == "experiment_v6"


def _write_data_yaml(path: Path, content: dict) -> None:
    path.write_text(yaml.dump(content), encoding="utf-8")


def test_validate_dataset_config_returns_false_when_missing(tmp_path):
    assert validate_dataset_config(tmp_path / "nope.yaml") is False


def test_validate_dataset_config_rejects_missing_required_keys(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    _write_data_yaml(yaml_path, {"path": str(tmp_path), "train": "train/images"})
    # missing val, nc, names
    assert validate_dataset_config(yaml_path) is False


def test_validate_dataset_config_rejects_missing_dirs(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    _write_data_yaml(yaml_path, {
        "path": str(tmp_path),
        "train": "train/images",
        "val": "val/images",
        "nc": 3,
        "names": ["a", "b", "c"],
    })
    # train/ and val/ dirs don't exist
    assert validate_dataset_config(yaml_path) is False


def test_validate_dataset_config_accepts_well_formed_yaml(tmp_path):
    train_dir = tmp_path / "train" / "images"
    val_dir = tmp_path / "val" / "images"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    (train_dir / "001.jpg").write_bytes(b"")
    (train_dir / "002.jpg").write_bytes(b"")
    (val_dir / "003.jpg").write_bytes(b"")

    yaml_path = tmp_path / "data.yaml"
    _write_data_yaml(yaml_path, {
        "path": str(tmp_path),
        "train": "train/images",
        "val": "val/images",
        "nc": 3,
        "names": ["leaf", "poop", "stick"],
    })
    assert validate_dataset_config(yaml_path) is True
