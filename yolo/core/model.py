from pathlib import Path
from typing import Any, List

import torch
import torch.nn as nn

from .builder import parse_model
from .utils import fuse_conv_and_bn, initialize_weights, load_weights, yaml_load
from .modules.conv import Conv as _Conv
from .modules.detect import Detect as _Detect


class SimplifiedDetectionModel(nn.Module):
    def __init__(self, 
                 cfg: str | Path, 
                 ch: int = 3, 
                 number_of_classes: int | None = None) -> None:
        super().__init__()
        self.yaml = yaml_load(cfg) if not isinstance(cfg, dict) else cfg
        if number_of_classes is not None:
            self.yaml["number_of_classes"] = number_of_classes
        self.number_of_classes = int(self.yaml.get("number_of_classes", 80))
        self.model, self.save = parse_model(self.yaml, [ch])
        initialize_weights(self)
        # Call bias_init on the Detect head for proper initialization
        last = self.model[-1]
        if isinstance(last, _Detect) and hasattr(last, "bias_init"):
            last.bias_init()

    def forward(self, 
                x: torch.Tensor) -> Any:
        outputs: List[torch.Tensor | None] = [None] * len(self.model)
        for i, module in enumerate(self.model):
            f = module.f if isinstance(module.f, list) else [module.f]
            if f == [-1]:
                input_tensor: torch.Tensor | List[torch.Tensor] = x
            else:
                tensors = []
                for j in f:
                    if j == -1:
                        tensors.append(x)
                    else:
                        tensors.append(outputs[j])
                input_tensor = tensors if len(tensors) > 1 else tensors[0]
            x = module(input_tensor)  
            outputs[i] = x
        return x

    def head_outputs(self, 
                     x: torch.Tensor) -> Any:
        return self.forward(x)

    def load(self, 
             weights: str | Path) -> None:
        load_weights(weights, self)

    def fuse(self) -> "SimplifiedDetectionModel":
        """Fold Conv+BatchNorm pairs into single convolutions for inference.

        Safe to call only in eval() mode. Mutates the model in place and
        also returns self for chaining.
        """
        for m in self.modules():
            if isinstance(m, _Conv) and isinstance(getattr(m, "bn", None), nn.BatchNorm2d):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                m.bn = nn.Identity()
                m.forward = m.forward_fuse
        return self


