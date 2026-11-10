from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2

from .core.logger import log_info
from .engine.checkpoint import load_model_from_checkpoint
from .engine.predictor import Predictor, draw_detections

IMG_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(source: str | Path) -> List[Path]:
    source = Path(source)
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_FORMATS)
    elif source.is_file() and source.suffix.lower() in IMG_FORMATS:
        images = [source]
    else:
        raise FileNotFoundError(f"Source must be an image file or a directory of images: {source}")
    if not images:
        raise FileNotFoundError(f"No images found in {source}")
    return images


def predict(weights: str | Path,
            source: str | Path,
            output: str | Path = "runs/predict/exp",
            imgsz: int | None = None,
            conf: float = 0.25,
            iou: float = 0.45,
            max_det: int = 300,
            device: str = "auto",
            use_ema: bool = True,
            save_txt: bool = False) -> Dict[str, int]:
    """Run detection on an image or folder, writing annotated images to ``output``.

    Returns a mapping of image filename to number of detections.
    """
    loaded = load_model_from_checkpoint(weights, device=device, use_ema=use_ema)
    predictor = Predictor(loaded.model, imgsz=imgsz or int(loaded.config.imgsz), conf=conf, iou=iou, max_det=max_det)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for path in collect_images(source):
        image = cv2.imread(str(path))
        if image is None:
            log_info("Skipping unreadable image %s", path)
            continue
        detections = predictor(image)
        cv2.imwrite(str(output / path.name), draw_detections(image, detections, loaded.names))
        if save_txt:
            h, w = image.shape[:2]
            lines = [
                f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f} {s:.4f}"
                for x1, y1, x2, y2, s, c in detections
            ]
            (output / f"{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        counts[path.name] = len(detections)
        summary: Dict[str, int] = {}
        for cls in detections[:, 5].astype(int):
            summary[loaded.names[cls]] = summary.get(loaded.names[cls], 0) + 1
        log_info("%s: %s", path.name, ", ".join(f"{n} x{k}" for n, k in summary.items()) or "no detections")
    log_info("Results saved to %s", output)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained checkpoint on images")
    parser.add_argument("--weights", required=True, help="Path to a trainer checkpoint (.pt)")
    parser.add_argument("--source", required=True, help="Image file or directory of images")
    parser.add_argument("--output", default="runs/predict/exp", help="Directory for annotated images")
    parser.add_argument("--imgsz", type=int, help="Inference size (default: the size used in training)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu' or e.g. 'cuda:0'")
    parser.add_argument("--no-ema", action="store_true", help="Use the raw weights instead of the EMA weights")
    parser.add_argument("--save-txt", action="store_true", help="Also write YOLO-format label files with confidence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict(
        args.weights,
        args.source,
        output=args.output,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        use_ema=not args.no_ema,
        save_txt=args.save_txt,
    )


if __name__ == "__main__":
    main()
