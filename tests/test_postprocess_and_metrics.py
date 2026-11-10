import pytest
import torch

from yolo.engine.metrics import mean_avg_precision_metric
from yolo.engine.postprocess import nms


def test_nms_suppresses_same_class_overlaps_only():
    # (batch, 4 + nc, anchors): rows are cx, cy, w, h, then per-class scores.
    pred = torch.tensor([[
        [50.0, 52.0, 50.0, 150.0],
        [50.0, 50.0, 50.0, 150.0],
        [40.0, 40.0, 40.0, 20.0],
        [40.0, 40.0, 40.0, 20.0],
        [0.9, 0.8, 0.0, 0.1],
        [0.0, 0.0, 0.7, 0.1],
    ]])
    out = nms(pred, conf_thres=0.25, iou_thres=0.5)[0]
    assert out.shape == (2, 6)
    by_class = {int(c): row for row, c in zip(out, out[:, 5])}
    assert by_class[0][:5].tolist() == pytest.approx([30, 30, 70, 70, 0.9])
    assert by_class[1][4].item() == pytest.approx(0.7)


def test_nms_returns_empty_when_nothing_passes_threshold():
    pred = torch.zeros(1, 6, 5)
    assert nms(pred, conf_thres=0.25)[0].shape == (0, 6)


# preds: [image, cls, conf, x1, y1, x2, y2]   targets: [image, cls, x1, y1, x2, y2]
TARGETS = torch.tensor([[0, 0, 10, 10, 50, 50], [1, 1, 20, 20, 80, 90]], dtype=torch.float32)


# 101-point interpolation caps a perfect AP at 0.995.
PERFECT_AP = 0.995


def test_map_is_maximal_for_perfect_predictions():
    preds = torch.tensor([[0, 0, 0.9, 10, 10, 50, 50], [1, 1, 0.8, 20, 20, 80, 90]], dtype=torch.float32)
    map50, map5095, _ = mean_avg_precision_metric(preds, TARGETS, num_classes=2)
    assert map50["map"] == pytest.approx(PERFECT_AP, abs=1e-3)
    assert map5095["map"] == pytest.approx(PERFECT_AP, abs=1e-3)


def test_map_is_zero_for_wrong_class_or_far_away_boxes():
    preds = torch.tensor([[0, 1, 0.9, 10, 10, 50, 50], [1, 1, 0.8, 200, 200, 240, 240]], dtype=torch.float32)
    map50, map5095, _ = mean_avg_precision_metric(preds, TARGETS, num_classes=2)
    assert map50["map"] == pytest.approx(0.0, abs=1e-6)
    assert map5095["map"] == pytest.approx(0.0, abs=1e-6)


def test_map5095_penalises_loose_boxes():
    # IoU of 0.82 passes the 0.5-0.8 thresholds but not 0.85-0.95.
    preds = torch.tensor([[0, 0, 0.9, 14, 10, 54, 50], [1, 1, 0.8, 20, 20, 80, 90]], dtype=torch.float32)
    map50, map5095, _ = mean_avg_precision_metric(preds, TARGETS, num_classes=2)
    assert map50["map"] == pytest.approx(PERFECT_AP, abs=1e-3)
    assert 0.7 < map5095["map"] < 0.95
