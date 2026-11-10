from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict

from .data.dataset import create_dataloaders
from .engine.checkpoint import load_model_from_checkpoint
from .engine.trainer import Trainer


def validate(weights: str | Path,
             data: str | Path | None = None,
             batch: int = 32,
             imgsz: int | None = None,
             device: str = "auto",
             workers: int = 4,
             use_ema: bool = True,
             per_class: bool = False) -> Dict[str, float]:
    """Evaluate a trainer checkpoint on the val split of a dataset yaml.

    Uses the same loss, NMS and mAP code as validation during training.
    """
    loaded = load_model_from_checkpoint(weights, device=device, use_ema=use_ema)
    overrides = {"device": device, "batch_size": batch, "workers": workers, "do_ema": False, "save": False}
    if data:
        overrides["data"] = str(data)
    if imgsz:
        overrides["imgsz"] = imgsz
    config = replace(loaded.config, **overrides)

    trainer = Trainer(loaded.model, config)
    _, trainer.val_loader = create_dataloaders(config, val_only=True)
    _, metrics = trainer.validate(config, trainer.val_loader)
    if per_class:
        trainer._log_per_class_metrics(metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint (loss, precision, recall, mAP)")
    parser.add_argument("--weights", required=True, help="Path to a trainer checkpoint (.pt)")
    parser.add_argument("--data", help="Dataset yaml with a 'val' split (default: the one used in training)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--imgsz", type=int, help="Validation image size (default: the size used in training)")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu' or e.g. 'cuda:0'")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--no-ema", action="store_true", help="Use the raw weights instead of the EMA weights")
    parser.add_argument("--per-class", action="store_true", help="Also print per-class precision, recall and mAP")
    parser.add_argument("--json", help="Write the metrics to this JSON file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = validate(
        args.weights,
        data=args.data,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        use_ema=not args.no_ema,
        per_class=args.per_class,
    )
    print(json.dumps(metrics, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
