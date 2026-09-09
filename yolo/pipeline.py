from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .config import SimplifiedConfig, load_config
from .engine.trainer import Trainer
from .model import build_model


class SimplifiedYoloPipeline:
    """Co-ordinates configuration, model creation, training and validation."""

    def __init__(self, config: SimplifiedConfig) -> None:
        self.config = config
        self.model = build_model(config)
        self._trainer: Optional[Trainer] = None

    def load_weights(self, weights_path: str | Path) -> None:
        """Load weights into the underlying YOLO model."""
        self.model.load(weights_path)

    def train(self, **extra_kwargs: Any) -> Dict[str, Any] | None:
        """Train the model with optional runtime overrides."""
        if extra_kwargs:
            self._apply_overrides(extra_kwargs)
        trainer = self._ensure_trainer()
        return trainer.fit()

    def validate(self, **extra_kwargs: Any) -> Dict[str, Any] | None:
        """Validate the model if validation is enabled."""
        if not self.config.validate and "data" not in extra_kwargs:
            return None
        if extra_kwargs:
            self._apply_overrides(extra_kwargs)
        trainer = self._ensure_trainer()
        return trainer.validate()

    def train_and_validate(self) -> Dict[str, Any] | None:
        """Train followed by validation when configured."""
        metrics = self.train()
        if self.config.validate:
            return self.validate()
        return metrics

    def _ensure_trainer(self) -> Trainer:
        if self._trainer is None:
            self._trainer = Trainer(self.model, self.config)
        return self._trainer

    def _apply_overrides(self, overrides: Mapping[str, Any]) -> None:
        reset_trainer = False
        for key, value in overrides.items():
            if key in {"epoch", "epochs"}:
                self.config.epochs = int(value)
                reset_trainer = True
            elif key in {"batch", "batch_size"}:
                self.config.batch_size = int(value)
                reset_trainer = True
            elif key in {"imgsz", "img_size"}:
                self.config.imgsz = int(value)
                reset_trainer = True
            elif key in {"pretrained", "weights"}:
                self.config.pretrained = str(value)
                self.model.load(str(value))
                reset_trainer = True
            elif hasattr(self.config, key):
                setattr(self.config, key, value)
                reset_trainer = True
            else:
                self.config.extra[key] = value
        if reset_trainer:
            self._trainer = None


def run_training(
    config_source: str | Path | Mapping[str, Any],
    *,
    overrides: Optional[Mapping[str, Any]] = None,
    validate: Optional[bool] = None,
) -> Dict[str, Any] | None:
    """Entry-point convenience wrapper used by CLI and scripts."""
    base_config = load_config(config_source)
    if overrides:
        config_dict = asdict(base_config)
        merged = {**config_dict, **dict(overrides)}
        if "extra" in overrides and isinstance(overrides["extra"], Mapping):
            merged_extra = {**config_dict.get("extra", {}), **overrides["extra"]}
            merged["extra"] = merged_extra
        valid_kwargs = {k: v for k, v in merged.items() if k in config_dict}
        base_config = replace(base_config, **valid_kwargs)
    if validate is not None:
        base_config = replace(base_config, validate=validate)

    pipeline = SimplifiedYoloPipeline(base_config)
    return pipeline.train_and_validate()
