import copy

import pytest
import torch

from yolo.core.model import SimplifiedDetectionModel
from yolo.core.utils import yaml_load

from .conftest import MODEL_YAML

# Parameter counts of the official Ultralytics YOLO11 models at nc=80 (unfused).
OFFICIAL_PARAMS = {"n": 2_624_080, "s": 9_458_752}


def build(scale: str, nc: int = 80) -> SimplifiedDetectionModel:
    cfg = copy.deepcopy(yaml_load(MODEL_YAML))
    cfg["scale"] = scale
    return SimplifiedDetectionModel(cfg, number_of_classes=nc)


@pytest.mark.parametrize("scale", sorted(OFFICIAL_PARAMS))
def test_parameter_count_matches_official_yolo11(scale):
    params = sum(p.numel() for p in build(scale).parameters())
    assert params == OFFICIAL_PARAMS[scale]


def test_train_mode_returns_one_feature_map_per_stride():
    model = build("n", nc=4).train()
    outs = model(torch.zeros(2, 3, 64, 64))
    no = 4 + 16 * 4
    assert [tuple(o.shape) for o in outs] == [(2, no, 8, 8), (2, no, 4, 4), (2, no, 2, 2)]


def test_eval_mode_returns_decoded_boxes_and_scores():
    model = build("n", nc=4).eval()
    decoded, feats = model(torch.zeros(1, 3, 64, 64))
    assert decoded.shape == (1, 4 + 4, 8 * 8 + 4 * 4 + 2 * 2)
    assert len(feats) == 3
    scores = decoded[:, 4:]
    assert scores.min() >= 0 and scores.max() <= 1


def test_default_conv_init_is_not_overridden():
    # Default init gives std ~0.017 on this layer; Kaiming fan_out would give ~0.042.
    model = build("n")
    assert model.model[5].conv.weight.std() < 0.03
