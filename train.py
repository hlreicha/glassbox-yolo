from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from yolo.pipeline import run_training

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified YOLO training entrypoint")
    parser.add_argument("--config", required=True, help="Path to a YAML/JSON config file")
    parser.add_argument("--weights", help="Optional path to initial weights")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--imgsz", type=int, help="Override image size")
    parser.add_argument("--device", help="Override training device, e.g. 'cuda:0'")
    parser.add_argument("--project", help="Override project directory")
    parser.add_argument("--name", help="Override run name")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation after training")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help="Resume training. Use bare flag to auto-load <project>/<name>/train/weights/last.pt, or pass an explicit checkpoint path.",
    )
    parser.add_argument("--extra", help="JSON string of additional overrides passed to YOLO.train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: Dict[str, Any] = {}
    if args.weights:
        overrides["pretrained"] = args.weights
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch_size"] = args.batch
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.device:
        overrides["device"] = args.device
    if args.project:
        overrides["project"] = args.project
    if args.name:
        overrides["name"] = args.name
    if args.resume is not None:
        overrides["resume"] = args.resume
    if args.extra:
        overrides.setdefault("extra", {}).update(json.loads(args.extra))

    metrics = run_training(args.config, overrides=overrides or None, validate=not args.no_validate)

    if metrics is not None:
        print(json.dumps(metrics, indent=2, default=lambda o: getattr(o, "__dict__", str(o))))


if __name__ == "__main__":
    main()
