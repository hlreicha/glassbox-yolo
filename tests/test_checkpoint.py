import copy
import sys
import types

import torch

from yolo import config as current_config
from yolo.core.model import SimplifiedDetectionModel
from yolo.core.utils import load_checkpoint_file, yaml_load
from yolo.engine.checkpoint import load_model_from_checkpoint

from .conftest import MODEL_YAML


def test_loads_checkpoint_pickled_under_old_package_name(tmp_path, tiny_config, monkeypatch):
    legacy = types.ModuleType("simplified_yolo")
    legacy.__path__ = []
    legacy.config = current_config
    monkeypatch.setitem(sys.modules, "simplified_yolo", legacy)
    monkeypatch.setitem(sys.modules, "simplified_yolo.config", current_config)
    path = tmp_path / "legacy.pt"
    monkeypatch.setattr(current_config.SimplifiedConfig, "__module__", "simplified_yolo.config")
    torch.save({"config": tiny_config}, path)
    monkeypatch.undo()

    assert "simplified_yolo" not in sys.modules
    ckpt = load_checkpoint_file(path)
    assert isinstance(ckpt["config"], current_config.SimplifiedConfig)
    assert ckpt["config"].imgsz == 64


def test_load_model_from_checkpoint(tiny_checkpoint):
    loaded = load_model_from_checkpoint(tiny_checkpoint, device="cpu")
    assert loaded.scale == "n"
    assert loaded.names == {0: "red", 1: "blue"}
    assert loaded.model.nc == 2
    assert not loaded.model.training


def test_scale_is_inferred_when_yaml_default_does_not_fit(tmp_path, tiny_config):
    cfg = copy.deepcopy(yaml_load(MODEL_YAML))
    cfg["scale"] = "s"
    model = SimplifiedDetectionModel(cfg, number_of_classes=2)
    path = tmp_path / "small.pt"
    torch.save({"model": model.state_dict(), "config": tiny_config, "names": {}}, path)

    loaded = load_model_from_checkpoint(path, device="cpu", use_ema=False)
    assert loaded.scale == "s"
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[key], value)
