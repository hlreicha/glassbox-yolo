from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml


@dataclass
class SimplifiedConfig:
    """Unified configuration for simplified YOLO training."""

    model: str
    data: str
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    workers: int = 8
    device: str = "auto"
    project: Optional[str] = None
    name: Optional[str] = None
    patience: Optional[int] = 50
    # resume: True -> auto-detect last.pt in save_dir; str -> explicit checkpoint path; False -> fresh start.
    resume: Any = False
    validate: bool = True
    save: bool = True
    pretrained: Optional[str] = None
    # When a pretrained checkpoint carries its own architecture spec, that spec
    # is used by default. Set this True to build from `model` (config yaml) instead.
    use_model_config: bool = False
    seed: Optional[int] = None
    lr0: Optional[float] = None
    lrf: Optional[float] = None
    cos_lr: bool = False
    weight_decay: Optional[float] = 5e-4
    momentum: float = 0.937
    optimizer: Optional[str] = None
    multi_scale: bool = False
    freeze: int = 0
    accumulate: int = 1
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    reference_batch_size: int = 64
    verbose: bool = True
    
    # EMA parameters
    do_ema: bool = True
    ema_decay: float = 0.9999
    ema_tau: float = 2000.0
    
    # Augmentation parameters
    mosaic: float = 1.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    flipud: float = 0.0
    fliplr: float = 0.5
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    rect: bool = False
    do_letterbox: bool = True
    close_mosaic: int = 10
    erasing: float = 0.0

    # Dataset filtering
    classes_to_ignore: Optional[list] = None
    single_class: bool = False

    # onnx params
    do_openvino: bool = False
    do_int8: bool = False
    metric_target: Optional[str] = "cpu"
    target: Optional[str] = "CPU"
    preset: str = "performance"
    conf_threshold: float = 0.4
    iou_threshold: float = 0.001
    ds_batch: int = 12
    export_onnx: bool = True
    export_opset: int = 12
    export_dynamic: bool = False
    export_simplify: bool = False
    export_ema: bool = False

    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def weights(self) -> Optional[str]:
        """Return the initial weights path if provided."""
        return self.pretrained
    
    def get_augmentation_hyp(self) -> Dict[str, float]:
        """Get augmentation hyperparameters as a dictionary."""
        return {
            'mosaic': self.mosaic,
            'do_letterbox': self.do_letterbox,
            'rect': self.rect,
            'hsv_h': self.hsv_h,
            'hsv_s': self.hsv_s,
            'hsv_v': self.hsv_v,
            'flipud': self.flipud,
            'fliplr': self.fliplr,
            'degrees': self.degrees,
            'translate': self.translate,
            'scale': self.scale,
            'shear': self.shear,
            'perspective': self.perspective,
            'erasing': self.erasing,
        }

    def to_train_kwargs(self) -> Dict[str, Any]:
        """Convert the configuration to keyword arguments accepted by ``YOLO.train``."""
        kwargs: Dict[str, Any] = {
            "data": self.data,
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.imgsz,
            "workers": self.workers,
            "device": self.device,
            "resume": self.resume,
            "multi_scale": self.multi_scale,
            "project": self.project,
            "name": self.name,
            "patience": self.patience,
            "save": self.save,
            "seed": self.seed,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "freeze": self.freeze,
            "accumulate": self.accumulate,
            "warmup_epochs": self.warmup_epochs,
            "verbose": self.verbose,
        }
        # remove ``None`` entries to keep the call clean
        return {k: v for k, v in {**kwargs, **self.extra}.items() if v is not None}

    def to_val_kwargs(self) -> Dict[str, Any]:
        """Subset of arguments for ``YOLO.val``."""
        return {
            "data": self.data,
            "imgsz": self.imgsz,
            "batch": self.batch_size,
            "device": self.device,
            "workers": self.workers,
        }



def load_config(source: str | Path | Mapping[str, Any]) -> SimplifiedConfig:
    """Load configuration from disk or dictionary.

    The loader accepts either a path to a JSON file or an in-memory mapping.
    Training configs must be JSON format. Dataset definitions (referenced in 'data' field) 
    can still be YAML format per YOLO standard.
    Nested sections with ``model`` and ``training`` keys are normalised into the flat
    ``SimplifiedConfig`` schema to keep the runtime lightweight.
    """

    raw = _load_raw(source)
    flat = _normalise(raw)
    return SimplifiedConfig(**flat)


def _load_raw(source: str | Path | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file '{path}' not found.")
    
    # Training configs must be JSON only
    if path.suffix != ".json":
        raise ValueError(
            f"Training config must be JSON format, got '{path.suffix}'. "
            f"Please use a .json file for training configuration. "
            f"(Dataset YAML files referenced in the 'data' field are still supported.)"
        )

    with path.open("r", encoding="utf-8") as handle:
        import json
        return json.load(handle)


def _normalise(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten hierarchical config dictionaries into ``SimplifiedConfig`` kwargs."""

    def pop(keys: Iterable[str], default: Any = None) -> Any:
        for key in keys:
            if key in raw_local:
                return raw_local.pop(key)
        return default

    raw_local = dict(raw)

    model_section = raw_local.pop("model", None)
    data_section = raw_local.pop("data", None)
    train_section = raw_local.pop("train", None)
    training_section = raw_local.pop("training", None)

    model_path: Optional[str] = None
    weights_path: Optional[str] = None

    if isinstance(model_section, Mapping):
        model_path = model_section.get("config") or model_section.get("path") or model_section.get("architecture")
        weights_path = model_section.get("weights") or model_section.get("checkpoint")
    elif isinstance(model_section, str):
        model_path = model_section

    if not model_path:
        model_path = pop(["model", "architecture", "config", "cfg"], None)

    if isinstance(data_section, Mapping):
        data_path = data_section.get("config") or data_section.get("path") or data_section.get("yaml")
    elif isinstance(data_section, str):
        data_path = data_section
    else:
        data_path = pop(["data", "dataset"], None)

    if not model_path or not data_path:
        missing = [name for name, value in (("model", model_path), ("data", data_path)) if not value]
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")

    merged = {
        "model": str(model_path),
        "data": str(data_path),
        **(_select_numeric(train_section) if isinstance(train_section, Mapping) else {}),
        **(_select_numeric(training_section) if isinstance(training_section, Mapping) else {}),
        **raw_local,
    }

    if weights_path is None:
        weights_path = merged.pop("weights", None) or merged.pop("checkpoint", None)
    else:
        # ensure we keep a reference to original weights key for backward compatibility
        merged.pop("weights", None)

    merged.setdefault("pretrained", weights_path)

    # rename possible aliases
    if "batch" in merged and "batch_size" not in merged:
        merged["batch_size"] = merged.pop("batch")
    if "imgsz" not in merged and "img_size" in merged:
        merged["imgsz"] = merged.pop("img_size")

    merged.setdefault("extra", {})

    return merged


def _select_numeric(section: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = {
        "epochs",
        "batch",
        "batch_size",
        "imgsz",
        "workers",
        "device",
        "patience",
        "resume",
        "validate",
        "save",
        "pretrained",
        "weights",
        "use_model_config",
        "seed",
        "lr0",
        "lrf",
        "weight_decay",
        "optimizer",
        "multi_scale",
        "freeze",
        "accumulate",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "reference_batch_size",
        "name",
        "project",
        "verbose",
    }
    return {k: v for k, v in section.items() if k in allowed}
