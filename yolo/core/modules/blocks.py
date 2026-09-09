'''
Hey you can build model blocks without excessively importing modules, how diabolical.
'''

from __future__ import annotations

from typing import Iterable
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, create_activation_function


__all__ = [
    "Bottleneck",
    "C3k",
    "C3k2",
    "SPPF",
    "Attention",
    "PSABlock",
    "liteMLA",
    "ABlock",
    "EBlock",
    "C2PSA",
]

class RepConv(nn.Module):
    """A convolutional block that combines two convolution layers (kernel and point-wise)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        activation: Optional[str] = "SiLU",
        **kwargs,
    ):
        super().__init__()
        self.act = create_activation_function(activation)
        self.conv1 = Conv(in_channels, out_channels, kernel_size, activation=False, **kwargs)
        self.conv2 = Conv(in_channels, out_channels, 1, activation=False, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv1(x) + self.conv2(x))


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        kernel_size: int | Iterable[int] = (3, 3),
        expansion: float = 0.5,
        do_repconv: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        k0, k1 = kernel_size
        middle_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, middle_channels, k0, 1) if not do_repconv else RepConv(in_channels, middle_channels, k0, activation=False)
        self.cv2 = Conv(middle_channels, out_channels, k1, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class RepNCSP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        groups: int = 1,
        dilation: int = 1,
        stride: int = 1,
        csp_expand: float = 0.5,
        repeat_num: int = 1,
        neck_args: Dict[str, Any] = {},
    ):
        neck_channels = int(out_channels * csp_expand)
        self.conv1 = Conv(in_channels, neck_channels, kernel_size=kernel_size, stride=stride, groups=groups, dilation=dilation)
        self.conv2 = Conv(in_channels, neck_channels, kernel_size=kernel_size, stride=stride, groups=groups, dilation=dilation)
        self.conv3 = Conv(2 * neck_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups, dilation=dilation)

        self.bottlenecks = nn.Sequential(
            *[Bottleneck(neck_channels, neck_channels,**neck_args) for _ in range(repeat_num)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.bottlenecks(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(torch.cat((y1, y2), dim=1))
    
class ELAN(nn.Module):
    """ELAN  structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        part_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        process_channels: Optional[int] = None,
    ):
        super().__init__()

        if process_channels is None:
            process_channels = part_channels // 2

        self.conv1 = Conv(in_channels, part_channels, kernel_size=1, stride=stride, groups=groups)
        self.conv2 = Conv(part_channels // 2, process_channels, kernel_size=3,stride=stride, groups=groups)
        self.conv3 = Conv(process_channels, process_channels, kernel_size=3, stride=stride, groups=groups)
        self.conv4 = Conv(part_channels + 2 * process_channels, out_channels, kernel_size=1, stride=stride, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(torch.cat([x1, x2, x3, x4], dim=1))
        return x5


class RepNCSPELAN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            part_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            groups: int = 1,
            process_channels: Optional[int] = None,
            csp_args: Dict[str, Any] = {},
            csp_neck_args: Dict[str, Any] = {}):
        super().__init__()

        if process_channels is None:
            process_channels = out_channels // 2
        
        self.conv1 = Conv(in_channels, process_channels, kernel_size=1, stride=stride, groups=groups)
        self.conv2 = nn.Sequential(
            RepNCSP(in_channels=part_channels // 2, process_channels=process_channels, neck_args=csp_neck_args, **csp_args),
            Conv(process_channels, process_channels, kernel_size=3, stride=1, groups=groups)
        )
        self.conv3 = nn.Sequential(
            RepNCSP(in_channels=process_channels, process_channels=process_channels, neck_args=csp_neck_args, **csp_args),
            Conv(process_channels, process_channels, kernel_size=3, stride=1, groups=groups)
        )
        self.conv4 = Conv(part_channels + 2 * process_channels, out_channels, kernel_size=1, stride=1, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)

        return self.conv4(torch.cat([x1, x2, x3, x4], dim=1))

    
class C3k(nn.Module):
    def __init__(self, in_channels: int, 
                 out_channels: int, 
                 repeats: int = 1, 
                 shortcut: bool = True, 
                 groups: int = 1, 
                 expansion: float = 0.5, 
                 kernel_size: int = 3) -> None:
        super().__init__()
        c_ = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(in_channels, c_, 1, 1)
        self.cv3 = Conv(2 * c_, out_channels, 1, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, groups, (kernel_size, kernel_size), expansion=1.0) for _ in range(repeats)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n: int = 1,
                 use_c3k: bool = False,
                 expand: float = 0.5,
                 groups: int = 1,
                 shortcut: bool = True,
                 ):
        super().__init__()
        self.hidden_dim = int(out_channels * expand)
        self.conv1 = Conv(in_channels, 2 * self.hidden_dim, kernel_size=1)
        self.conv2 = Conv((2 + n) * self.hidden_dim, out_channels, kernel_size=1)



        self.bottleneck = nn.ModuleList(C3k(self.hidden_dim,self.hidden_dim,repeats=2,shortcut=shortcut,groups=groups) if use_c3k else Bottleneck(self.hidden_dim,self.hidden_dim,shortcut=shortcut,expansion=0.5,groups=groups) for _ in range(n))
    
    def forward(self, x):
        y = list(self.conv1(x).chunk(2,1))
        y.extend(m(y[-1]) for m in self.bottleneck)
        
        return self.conv2(torch.cat(y, dim=1))

class SPPF(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 5) -> None:
        super().__init__()
        c_ = in_channels // 2
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, out_channels, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        for _ in range(3):
            y.append(self.pool(y[-1]))
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    def __init__(self, 
                 dim: int, 
                 num_heads: int = 8, 
                 attn_ratio: float = 0.5) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = int(dim // num_heads)
        self.key_dim = max(int(self.head_dim * attn_ratio), 1)
        key_dim_n = self.key_dim * num_heads
        self.normalize_scale = self.key_dim ** -0.5
        self.qkv = Conv(dim, dim + 2 * key_dim_n, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, groups=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.normalize_scale
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).view(b, c, h, w)
        return self.proj(out + self.pe(v.reshape(b, c, h, w)))

        


class PSABlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 attn_ratio: float = 0.5, 
                 num_heads: int = 4, 
                 expansion: float = 2.0,
                 shortcut: bool = True) -> None:
        super().__init__()
        self.attn = Attention(in_channels, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(Conv(in_channels, int(in_channels * expansion), 1), Conv(int(in_channels * expansion), in_channels, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class AConv(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=3,stride=1)
        self.conv = Conv(in_channels,out_channels,kernel_size = 3, stride = 2)
    def forward(self,x):
        return self.conv(self.avg_pool(x))
    
class RepNCSP(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kenel_size: int = 1,
            repeats: int = 1,
            expansion: float = 0.5,
            groups: int = 1,
            shortcut: bool = True,
            neck_args: Dict[str, Any] = {}
    ):
        super().__init__()
        neck_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, neck_channels, kernel_size=kenel_size, groups=groups)
        self.conv2 = Conv(in_channels, neck_channels, kernel_size=kenel_size, groups=groups)
        self.conv3 = Conv(2 * neck_channels, out_channels, kernel_size=kenel_size, groups=groups)

        self.bottlenecks = nn.Sequential(
            *[Bottleneck(neck_channels, neck_channels, shortcut, groups, (3, 3), **neck_args) for _ in range(repeats)]
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.bottlenecks(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(torch.cat((y1, y2), dim=1))
    
class C2PSA(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 repeats: int = 1, 
                 expansion: float = 0.5) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("out_channelsPSA expects equal input/output channels")
        self.c = int(in_channels * expansion)
        self.cv1 = Conv(in_channels, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, in_channels, 1, 1)
        heads = max(1, self.c // 64)
        self.psa_blocks = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=heads) for _ in range(repeats)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.psa_blocks(b)
        return self.cv2(torch.cat((a, b), 1))

class CBLinear(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels_list: List[int],
                 kernel_size: int = 1,
                 stride: int = 1,
                 groups: int = 1,):
        super().__init__()
        self.out_channels_list = out_channels_list
        self.conv = Conv(in_channels, sum(out_channels_list), kernel_size=kernel_size, stride=stride, groups=groups,act=False,bias=True,bn = False)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        outs = torch.split(x, self.out_channels_list, dim=1)
        return outs

class CBFuse(nn.Module):
    def __init__(self,
                 index: List[int],
                 mode: str = 'nearest'):
        super().__init__()
        self.index = index
        self.mode = mode

    def forward(self,
                x_list: List[torch.Tensor]) -> torch.Tensor:
        target = x_list[-1]
        target_size = target.shape[2:] 
        res = [F.interpolate(x[pick_id], size=target_size, mode=self.mode) for pick_id, x in zip(self.index, x_list)]
        out = torch.stack(res + [target]).sum(dim=0)
        return out



class liteMLA(nn.Module):
    def __init__(
        self,
        dim: int = 32,
        num_heads: int = 4,
        area: int = 4,
        scales: tuple[int, int] = (5, 3),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.area = area
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scales = scales
        self.eps = eps
        all_head_dim = self.head_dim * self.num_heads
        self.qkv = Conv(dim, all_head_dim, 1, 1, d=2, act=False)
        self.aggregations = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(all_head_dim, all_head_dim, scale, padding=scale // 2, groups=all_head_dim, bias=False),
                nn.Conv2d(all_head_dim, all_head_dim, 1, groups=num_heads, bias=False),
            )
            for scale in scales
        )
        self.final = Conv(all_head_dim, dim, 1)
        self.positional = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False, bias=True)

    @torch.autocast(device_type="cuda", enabled=False)
    def _relu_linear_attention(self, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = qkv.shape
        n = h * w
        if qkv.dtype in (torch.float16, torch.bfloat16):
            qkv = qkv.float()
        qkv = qkv.reshape(b, -1, n)
        if self.area > 1:
            qkv = qkv.reshape(b * self.area, 3 * self.dim, n // self.area)
            b, _, n = qkv.shape
        q, k, v = qkv.view(b, self.num_heads, self.head_dim * 3, n).split(
            [self.head_dim, self.head_dim, self.head_dim], dim=2
        )
        v_orig = v
        q = F.relu(q)
        k = F.relu(k)
        prod = torch.matmul(F.pad(v, (0, 0, 0, 1), value=1.0), k.transpose(-2, -1))
        attn = torch.matmul(prod, q)
        attn = attn[:, :, :-1] / attn[:, :, -1:].clamp_min(self.eps)
        if self.area > 1:
            attn = attn.reshape(b // self.area, self.dim, n * self.area)
            v_orig = v_orig.reshape(b // self.area, self.dim, n * self.area)
            b, _, n = attn.shape
        attn = attn.reshape(b, -1, h, w)
        v_orig = v_orig.reshape(b, -1, h, w)
        return attn, v_orig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.qkv(x)
        multi_scale = [base] + [layer(base) for layer in self.aggregations]
        qkv = torch.cat(multi_scale, dim=1)
        attn, v_orig = self._relu_linear_attention(qkv)
        return self.final(attn + self.positional(v_orig))


class ABlock(nn.Module):
    def __init__(self, 
                 dim: int, 
                 num_heads: int, 
                 expansion: float = 1.2, 
                 area: int = 1) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.attn = liteMLA(dim, num_heads=num_heads, area=area)
        self.mlp = nn.Sequential(Conv(dim, hidden, 1), Conv(hidden, dim, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x)


class EBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.attn = liteMLA(dim, num_heads=num_heads, area=area)
        self.mlp = nn.Sequential(Conv(dim, hidden, 1), Conv(hidden, dim, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x)


