from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from yolo.config import SimplifiedConfig
from yolo.core.model import SimplifiedDetectionModel

REPO = Path(__file__).resolve().parents[1]
MODEL_YAML = REPO / "yolo" / "model_config" / "yolo11.yaml"
NC = 2


def _write_split(root: Path, split: str, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    images = root / "images" / split
    labels = root / "labels" / split
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in range(n):
        img = rng.integers(0, 60, (64, 64, 3), dtype=np.uint8)
        cls = i % NC
        x, y = rng.integers(4, 24, 2)
        w, h = rng.integers(20, 36, 2)
        img[y:y + h, x:x + w] = (0, 0, 255) if cls == 0 else (255, 0, 0)
        cv2.imwrite(str(images / f"{split}_{i}.png"), img)
        (labels / f"{split}_{i}.txt").write_text(
            f"{cls} {(x + w / 2) / 64:.6f} {(y + h / 2) / 64:.6f} {w / 64:.6f} {h / 64:.6f}\n"
        )


@pytest.fixture
def tiny_dataset(tmp_path) -> Path:
    root = tmp_path / "data"
    _write_split(root, "train", 16, seed=0)
    _write_split(root, "val", 8, seed=1)
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "red", 1: "blue"},
    }))
    return data_yaml


@pytest.fixture
def tiny_config(tiny_dataset) -> SimplifiedConfig:
    return SimplifiedConfig(
        model=str(MODEL_YAML),
        data=str(tiny_dataset),
        imgsz=64,
        batch_size=4,
        workers=0,
        device="cpu",
        epochs=1,
        extra={"nc": NC},
    )


@pytest.fixture
def tiny_checkpoint(tmp_path, tiny_config) -> Path:
    """A randomly initialised nc=2 checkpoint in the trainer's on-disk format."""
    model = SimplifiedDetectionModel(str(MODEL_YAML), number_of_classes=NC)
    path = tmp_path / "tiny.pt"
    torch.save({
        "epoch": 1,
        "model": model.state_dict(),
        "ema": model.state_dict(),
        "config": tiny_config,
        "names": {0: "red", 1: "blue"},
    }, path)
    return path
