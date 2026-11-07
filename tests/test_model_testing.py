import time
from pathlib import Path

import pytest

from model_testing import (
    Detection,
    calculate_iou,
    find_latest_model,
    remove_duplicate_detections,
)


# calculate_iou ---------------------------------------------------------------

def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert calculate_iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert calculate_iou((0.0, 0.0, 5.0, 5.0), (10.0, 10.0, 20.0, 20.0)) == 0.0


def test_iou_touching_edge_is_zero():
    # Boxes share only an edge; intersection area is 0
    assert calculate_iou((0.0, 0.0, 5.0, 5.0), (5.0, 0.0, 10.0, 5.0)) == 0.0


def test_iou_half_overlap():
    # 10x10 box and 10x10 box offset by 5 in x -> intersection 5x10=50, union 100+100-50=150
    assert calculate_iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0)) == pytest.approx(50 / 150)


def test_iou_box_fully_inside_another():
    outer = (0.0, 0.0, 10.0, 10.0)
    inner = (2.0, 2.0, 4.0, 4.0)
    # intersection = inner area = 4; union = outer area = 100; iou = 4/100
    assert calculate_iou(outer, inner) == pytest.approx(4 / 100)


# remove_duplicate_detections -------------------------------------------------

def _det(conf: float, bbox) -> Detection:
    return Detection(confidence=conf, class_id=0, class_name="x", bbox=bbox)


def test_remove_duplicates_empty_list():
    assert remove_duplicate_detections([]) == []


def test_remove_duplicates_single_detection_unchanged():
    d = _det(0.9, (0.0, 0.0, 10.0, 10.0))
    assert remove_duplicate_detections([d]) == [d]


def test_remove_duplicates_keeps_higher_confidence_of_overlapping_pair():
    high = _det(0.9, (0.0, 0.0, 10.0, 10.0))
    low = _det(0.4, (1.0, 1.0, 11.0, 11.0))  # ~81% iou with high
    kept = remove_duplicate_detections([low, high])
    assert kept == [high]


def test_remove_duplicates_keeps_both_when_disjoint():
    a = _det(0.9, (0.0, 0.0, 10.0, 10.0))
    b = _det(0.8, (100.0, 100.0, 110.0, 110.0))
    kept = remove_duplicate_detections([a, b])
    assert set(id(d) for d in kept) == {id(a), id(b)}


def test_remove_duplicates_three_overlapping_keeps_highest():
    # Boxes with small offsets so all three IoU > 0.5 against the first
    a = _det(0.9, (0.0, 0.0, 10.0, 10.0))
    b = _det(0.7, (0.5, 0.5, 10.5, 10.5))
    c = _det(0.5, (0.5, 0.5, 10.5, 10.5))
    assert remove_duplicate_detections([a, b, c]) == [a]


# find_latest_model -----------------------------------------------------------

def test_find_latest_model_returns_none_when_dir_missing(tmp_path):
    assert find_latest_model(tmp_path / "does_not_exist") is None


def test_find_latest_model_returns_none_when_no_weights(tmp_path):
    # directory exists but has no matching pattern
    (tmp_path / "poop_detector_v1").mkdir()
    assert find_latest_model(tmp_path) is None


def test_find_latest_model_finds_single_weights(tmp_path):
    weights_dir = tmp_path / "poop_detector_v1" / "weights"
    weights_dir.mkdir(parents=True)
    best = weights_dir / "best.pt"
    best.write_bytes(b"")
    assert find_latest_model(tmp_path) == best


def test_find_latest_model_returns_newest_by_mtime(tmp_path):
    older = tmp_path / "poop_detector_v1" / "weights"
    newer = tmp_path / "poop_detector_v2" / "weights"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    older_pt = older / "best.pt"
    newer_pt = newer / "best.pt"
    older_pt.write_bytes(b"")
    # Give the filesystem a measurable mtime difference
    time.sleep(0.01)
    newer_pt.write_bytes(b"")
    assert find_latest_model(tmp_path) == newer_pt
