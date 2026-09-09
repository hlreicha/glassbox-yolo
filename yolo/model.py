from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import SimplifiedConfig
from .core.logger import log_info
from .core.model import SimplifiedDetectionModel
from .core.utils import apply_state_dict, read_checkpoint


def build_model(config: SimplifiedConfig, *, weights: Optional[str] = None) -> SimplifiedDetectionModel:
    """Instantiate the simplified detection model from configuration.

    When a pretrained checkpoint is supplied and carries its own architecture
    spec, that spec is used to build the model. Set config.use_model_config=True
    to force construction from config.model instead.
    """
    weight_source = weights or config.weights
    nc = config.extra.get("nc") if hasattr(config, "extra") else None

    state_dict: Optional[dict] = None
    model_reference: str | Path | dict = Path(config.model)

    if weight_source:
        ckpt_yaml = None
        try:
            state_dict, ckpt_yaml = read_checkpoint(weight_source)
        except Exception as e:
            log_info("Failed to read checkpoint '%s': %s. Building from '%s'.", weight_source, e, config.model)
            state_dict = None

        force_config_arch = bool(getattr(config, "use_model_config", False))
        if ckpt_yaml is not None and not force_config_arch:
            if nc is not None:
                ckpt_yaml["nc"] = nc
            model_reference = ckpt_yaml
            log_info("Using architecture from checkpoint '%s'.", weight_source)
        elif ckpt_yaml is not None and force_config_arch:
            log_info("use_model_config=True, ignoring checkpoint architecture and building from '%s'.", config.model)

    detection_model = SimplifiedDetectionModel(model_reference, number_of_classes=nc)

    if state_dict is not None:
        apply_state_dict(state_dict, detection_model)

    return detection_model