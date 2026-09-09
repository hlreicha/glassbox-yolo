import math

import torch
import torch.nn as nn

__all__ = ["autopad", "Conv", "DWConv", "Concat"]


def create_activation_function(activation: str | bool) -> nn.Module:
    """
    Retrieves an activation function from the PyTorch nn module based on its name, case-insensitively.
    """
    if not activation or (isinstance(activation, str) and activation.lower() in ["false", "none"]):
        return nn.Identity()
    
    if activation is True:
        return nn.SiLU(inplace=True)

    activation_map = {
        name.lower(): obj
        for name, obj in nn.modules.activation.__dict__.items()
        if isinstance(obj, type) and issubclass(obj, nn.Module)
    }
    if activation.lower() in activation_map:
        return activation_map[activation.lower()](inplace=True)
    else:
        raise ValueError(f"Activation function '{activation}' is not found in torch.nn")

def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        dilation: int = 1,
        act: bool | str = True,
        bias: bool | None = None,
        bn: bool = True,
    ) -> None:
        super().__init__()
        if bias is None:
            bias = False
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding, dilation), groups=groups, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=1e-3) if bn else nn.Identity()
        self.act = create_activation_function(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class DWConv(Conv):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, dilation: int = 1, act: bool | str = True) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=None, groups=math.gcd(in_channels, out_channels), dilation=dilation, act=act)


class Concat(nn.Module):
    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.d = dimension

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, self.d)
