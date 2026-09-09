from copy import deepcopy
from importlib import import_module
from typing import Any, Dict, List, Tuple

import torch.nn as nn

from .modules import (
    C2PSA,
    C3k,
    C3k2,
    Concat,
    Conv,
    Detect,
    SPPF,
)
from .utils import make_divisible

MODULE_REGISTRY = {
    "Conv": Conv,
    "C3k": C3k,
    "C3k2": C3k2,
    "SPPF": SPPF,
    "C2PSA": C2PSA,
    "Concat": Concat,
    "Detect": Detect,
}


class LayerSpec:
    def __init__(self, from_idx: Any, repeats: int, module: Any, args: List[Any]) -> None:
        self.from_idx = from_idx
        self.repeats = repeats
        self.module = module
        self.args = args


def _resolve_module(name: str) -> Any:
    if name in MODULE_REGISTRY:
        return MODULE_REGISTRY[name]
    if name.startswith("nn."):
        attr = name.split(".", 1)[1]
        return getattr(nn, attr)
    module_name, attr = name.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr)

def round_channels(value: int,width_multiple: float, max_channels: int) -> int:
    scaled = make_divisible(value * width_multiple, 8)
    return int(min(max_channels, max(int(scaled), 1)))

def round_repeats(value: int, depth_multiple: float) -> int:
    if value <= 1:
        return int(value)
    return max(int(round(value * depth_multiple)), 1)

def get_channel(index: int, channels: List[int]) -> int:
    if index == -1:
        if not channels:
            raise IndexError("Channel list is empty when resolving -1")
        return channels[-1]
    if index < 0:
        # support negative indices other than -1 (e.g., -2, -3)
        if len(channels) + index < 0:
            raise IndexError(f"Channel index {index} out of range for channels list of size {len(channels)}")
        return channels[index]
    if index >= len(channels):
        raise IndexError(f"Channel index {index} out of range for channels list of size {len(channels)}")
    return channels[index]


def parse_model(cfg: Dict[str, Any], ch: List[int] | int) -> Tuple[nn.ModuleList, List[int]]:
    yaml = deepcopy(cfg)
    layers: List[nn.Module] = []
    save: List[int] = []

    scales = yaml.get("scales", {}) if isinstance(yaml.get("scales"), dict) else {}
    
    if "scale" not in yaml:
        scale_key = "n"
        print("Scale key must be specified in the configuration or scales dictionary, defaulting to 'n'")
    else:
        scale_key = yaml.get("scale")


    depth_multiple = float(yaml.get("depth_multiple", 1.0))
    width_multiple = float(yaml.get("width_multiple", 1.0))
    max_channels = int(yaml.get("max_channels", 1024))
    if scale_key and scale_key in scales:
        depth_multiple, width_multiple, max_channels = scales[scale_key]
    elif not scale_key and scales:
        depth_multiple, width_multiple, max_channels = next(iter(scales.values()))

    yaml["depth_multiple"] = depth_multiple
    yaml["width_multiple"] = width_multiple
    yaml["max_channels"] = max_channels



    if isinstance(ch, (list, tuple)):
        channels = [int(c) for c in ch]
    else:
        channels = [int(ch)]
    if not channels:
        raise ValueError("At least one input channel must be provided")

    model_config = yaml.get("model",{})
    assert model_config is not {}, "Model configuration is None or Empty, please pass model configuration..."
    
    module_defs = model_config.get("backbone", []) + model_config.get("head", []) + model_config.get("aux", [])
    if not module_defs:
        raise ValueError("Model configuration missing backbone/head definitions")

    for i, module_dict in enumerate(module_defs):
        module_name = list(module_dict.keys())[0]
        from_idx = module_dict[module_name]['from_idx']
        num_of_repeats = module_dict[module_name]['num_of_repeats']
        args = module_dict[module_name]['args']

        module_cls = _resolve_module(module_name)
        args = deepcopy(args)
        args = [None if isinstance(a, str) and a.lower() == "none" else a for a in args]

        f = from_idx if isinstance(from_idx, list) else [from_idx]
        num_of_repeats = round_repeats(int(num_of_repeats), depth_multiple)

        out_channels: int | None = None

        if module_cls in {Conv, SPPF, C3k, C3k2, C2PSA}:
            in_channels = get_channel(f[0], channels)
            out = round_channels(int(args[0]), width_multiple, max_channels)
            args = [in_channels, out, *args[1:]]
            if module_cls in {C3k, C3k2, C2PSA}:
                args.insert(2, num_of_repeats)
                num_of_repeats = 1
            out_channels = out
        elif module_cls is Concat:
            out_channels = sum(get_channel(x, channels) for x in f)
        elif module_cls is Detect:
            ch_list = [get_channel(x, channels) for x in f]
            args = [yaml.get("number_of_classes"), ch_list]
            out_channels = yaml.get("number_of_classes", 80) + 5
        elif module_cls is nn.Upsample:
            args = [*args]
            out_channels = get_channel(f[0], channels)
        else:
            in_channels = get_channel(f[0], channels)
            args = [in_channels, *args]
            if hasattr(module_cls, "__name__") and module_cls.__name__ == "BatchNorm2d":
                out_channels = in_channels
            else:
                out_channels = in_channels

        if num_of_repeats > 1 and module_cls not in {Concat, Detect}:
            module = nn.Sequential(*(module_cls(*args) for _ in range(num_of_repeats)))
        else:
            module = module_cls(*args)

        module.i = i 
        module.f = from_idx  
        module.type = module.__class__.__name__  
        module.n_channels = out_channels  

        layers.append(module)
        if i == 0 and len(channels) == 1:
            channels = []
        if out_channels is None:
            raise ValueError(f"Could not determine output channels for layer {i} ({module_name})")
        channels.append(out_channels)

        for x in (f if isinstance(f, list) else [f]):
            if x != -1 and i > 0:
                save.append(x % i)

    return nn.ModuleList(layers), sorted(set(save))