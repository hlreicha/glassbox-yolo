from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from ..core.logger import log_info
from ..core.model import SimplifiedDetectionModel
from ..core.utils import load_checkpoint_file, yaml_load
from .export import resolve_model_cfg


@dataclass
class LoadedModel:
    model: SimplifiedDetectionModel
    config: Any
    names: Dict[int, str]
    scale: str | None


def resolve_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def _matches(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> bool:
    own = model.state_dict()
    return own.keys() == state_dict.keys() and all(own[k].shape == v.shape for k, v in state_dict.items())


def load_model_from_checkpoint(weights: str | Path,
                               device: str = "cpu",
                               use_ema: bool = True,
                               model_cfg: str | Path | None = None) -> LoadedModel:
    """Rebuild a trained model from a trainer checkpoint.

    The checkpoint does not record which scale (n/s/m/l/x) of the model yaml was
    trained, so every scale in the yaml is tried until the weights fit.
    """
    ckpt = load_checkpoint_file(weights)
    if not isinstance(ckpt, dict) or "config" not in ckpt:
        raise ValueError(f"{weights} was not written by this repo's trainer (no 'config' entry)")
    config = ckpt["config"]
    key = "ema" if use_ema and "ema" in ckpt else "model"
    state_dict = ckpt[key]

    cfg_path = resolve_model_cfg(model_cfg or config.model)
    cfg = yaml_load(cfg_path)
    nc = config.extra.get("nc")

    preferred = cfg.get("scale")
    scales = [preferred] + [s for s in (cfg.get("scales") or {}) if s != preferred]
    for scale in scales:
        trial = deepcopy(cfg)
        if scale is not None:
            trial["scale"] = scale
        model = SimplifiedDetectionModel(trial, number_of_classes=nc)
        if _matches(model, state_dict):
            break
    else:
        raise ValueError(f"Weights in {weights} do not fit any scale of {cfg_path} (tried {scales})")
    if scale != preferred:
        log_info("Checkpoint matches scale '%s' of %s (yaml default is '%s')", scale, cfg_path, preferred)

    model.load_state_dict(state_dict)
    model.nc = model.number_of_classes
    names = {int(k): str(v) for k, v in (ckpt.get("names") or {}).items()}
    names = {i: names.get(i, str(i)) for i in range(model.nc)}
    model.names = names
    model.to(resolve_device(device)).float().eval()
    return LoadedModel(model=model, config=config, names=names, scale=scale)
