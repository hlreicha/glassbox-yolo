import cv2
import numpy as np
import torch
from torch import nn

from yolo.engine.predictor import Predictor, letterbox_image
from yolo.predict import collect_images, predict
from yolo.val import validate


class FixedDetector(nn.Module):
    """Returns one fixed detection (cx, cy, w, h in input pixels, then class scores)."""

    def __init__(self, box, scores):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.out = torch.tensor([*box, *scores]).view(1, -1, 1)

    def forward(self, x):
        return self.out


def test_letterbox_centres_and_preserves_aspect_ratio():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    canvas, ratio, left, top = letterbox_image(image, 64)
    assert canvas.shape == (64, 64, 3)
    assert ratio == 0.32
    assert (left, top) == (0, 16)


def test_predictor_maps_boxes_back_to_original_pixels():
    # Original box (50, 20, 150, 80) in a 200x100 image, seen through a 64px letterbox.
    detector = FixedDetector(box=(32, 32, 32, 19.2), scores=(0.9, 0.0))
    det = Predictor(detector, imgsz=64, conf=0.25)(np.zeros((100, 200, 3), dtype=np.uint8))
    assert det.shape == (1, 6)
    np.testing.assert_allclose(det[0, :4], [50, 20, 150, 80], atol=0.6)
    assert det[0, 5] == 0


def test_predict_writes_annotated_image_and_labels(tiny_checkpoint, tmp_path):
    image = np.random.default_rng(0).integers(0, 255, (100, 150, 3), dtype=np.uint8)
    source = tmp_path / "in"
    source.mkdir()
    cv2.imwrite(str(source / "a.png"), image)

    counts = predict(tiny_checkpoint, source, output=tmp_path / "out", conf=0.0, device="cpu", save_txt=True)

    assert counts["a.png"] > 0
    assert cv2.imread(str(tmp_path / "out" / "a.png")).shape == image.shape
    assert len((tmp_path / "out" / "a.txt").read_text().splitlines()) == counts["a.png"]
    assert collect_images(source) == [source / "a.png"]


def test_validate_reports_metrics(tiny_checkpoint, tiny_dataset):
    metrics = validate(tiny_checkpoint, data=tiny_dataset, batch=4, device="cpu", workers=0)
    assert {"val_loss", "precision", "recall", "mAP50", "mAP50-95", "fitness"} <= metrics.keys()
    assert all(np.isfinite(v) for v in metrics.values())
