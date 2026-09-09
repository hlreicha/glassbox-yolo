import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import yaml

from .logger import log_info


def yaml_load(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def make_divisible(v: float, divisor: int) -> int:
    """Return the nearest value that is divisible by ``divisor``."""
    return int(math.ceil(v / divisor) * divisor)


def initialize_weights(module: nn.Module) -> None:
    """Initialize model weights following PyTorch defaults.

    Skips frozen parameters so that special-purpose layers (like the DFL
    integration conv, which is a fixed arange and has requires_grad=False)
    are not clobbered.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            if m.weight.requires_grad is False:
                continue
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def intersect_dicts(source: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the subset of source entries whose keys and shapes match target.
    Entries with mismatched shapes are dropped so the model keeps its init values.
    """
    result: Dict[str, Any] = {}
    skipped_shape = 0
    for k, v in source.items():
        if k not in target:
            continue
        if not (isinstance(v, torch.Tensor) and isinstance(target[k], torch.Tensor)):
            continue
        if v.shape == target[k].shape:
            result[k] = v
        else:
            skipped_shape += 1
    if skipped_shape:
        log_info("Skipping %d weight entries due to shape mismatch", skipped_shape)
    return result


def read_checkpoint(path: str | Path) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Load a checkpoint file and return (state_dict, model_yaml).

    Handles Ultralytics-style .pt files that pickle an nn.Module under 'ema' or
    'model', Trainer-style dicts with a 'model' state_dict entry, and plain
    state_dict files. model_yaml is None when the checkpoint does not carry an
    architecture spec.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        entry = ckpt.get("ema") or ckpt.get("model") or ckpt
    else:
        entry = ckpt

    model_yaml = getattr(entry, "yaml", None)
    if not isinstance(model_yaml, dict):
        model_yaml = None

    if isinstance(entry, dict):
        state_dict = entry
    else:
        state_dict = entry.state_dict()

    return state_dict, model_yaml


def apply_state_dict(state_dict: Mapping[str, Any], module: nn.Module) -> None:
    """Copy shape-compatible parameters from state_dict into module."""
    filtered = intersect_dicts(state_dict, module.state_dict())
    missing = set(module.state_dict().keys()) - set(filtered.keys())
    if missing:
        log_info("Skipping %d unmatched parameters during weight load", len(missing))
    module.load_state_dict(filtered, strict=False)


def load_weights(path: str | Path, module: nn.Module) -> None:
    """Load a checkpoint file into module. Shape-incompatible entries are skipped."""
    state_dict, _ = read_checkpoint(path)
    apply_state_dict(state_dict, module)

def count_parameters(module: nn.Module) -> int:
    """Count the total number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def ensure_list(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return [value]

def fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse convolution and batch normalization layers into a single convolution."""
    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels = conv.out_channels,
        kernel_size = conv.kernel_size,
        stride = conv.stride,
        padding = conv.padding,
        dilation = conv.dilation,
        groups = conv.groups,
        bias = True).requires_grad_(False).to(conv.weight.device)
    conv_reshaped = conv.weight.view(conv.out_channels,-1)
    scales = bn.weight.div(torch.sqrt(bn.running_var + bn.eps))
    new_conv.weight.copy_((conv_reshaped * scales[:,None]).view(conv.weight.shape))
    b_conv = torch.zeros(conv.out_channels, device=conv.weight.device) if conv.bias is None else conv.bias
    new_conv.bias.copy_(scales * (b_conv - bn.running_mean) + bn.bias)

    return new_conv

def fuse_deconv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse dw_convolution and batch normalization layers into a single convolution."""
    new_conv = nn.ConvTranspose2d(
        in_channels=conv.in_channels,
        out_channels = conv.out_channels,
        kernel_size = conv.kernel_size,
        stride = conv.stride,
        padding = conv.padding,
        dilation = conv.dilation,
        groups = conv.groups,
        bias = True).requires_grad_(False).to(conv.weight.device)
    
    conv_reshaped = conv.weight.view(conv.out_channels,-1)
    scales = bn.weight.div(torch.sqrt(bn.running_var + bn.eps))
    new_conv.weight.copy_((conv_reshaped * scales[:,None]).view(conv.weight.shape))
    b_conv = torch.zeros(conv.out_channels, device=conv.weight.device) if conv.bias is None else conv.bias
    new_conv.bias.copy_(scales * (b_conv - bn.running_mean) + bn.bias)

    return new_conv
