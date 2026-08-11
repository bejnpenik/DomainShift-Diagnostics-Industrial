import torch
import torch.nn as nn
import torch.optim as optim


class Conv1D(nn.Module):
    def __init__(self, input:int, output:int, k:int, s:int, p:int=0, d:int=1, g:int=1, act:nn.Module=nn.ReLU(), bn:bool=False):
        super().__init__()
        self._conv = nn.Conv1d(input, output, k, s, p, d, g)
        self._act = act
        self._bn = nn.BatchNorm1d(output) if bn else nn.Identity()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self._act(self._bn(self._conv(x)))
    

class Conv2D(nn.Module):
    def __init__(self, input:int, output:int, k:int|tuple, s:int|tuple, p:int|tuple, d:int=1, g:int=1, act:nn.Module=nn.ReLU(), bn:bool=False):
        super().__init__()
        self._conv = nn.Conv2d(input, output, k, s, p, d, g)
        self._act = act
        self._bn = nn.BatchNorm2d(output) if bn else nn.Identity()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self._act(self._bn(self._conv(x)))
    
class Pool1D(nn.Module):
    def __init__(self, k:int, s:int, p:int=0, d:int=1, maxpool:bool=True):
        super().__init__()
        self._pool = nn.MaxPool1d(k, s, p, d) if maxpool else nn.AvgPool1d(k, s, p, d)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self._pool(x)

class Pool2D(nn.Module):
    def __init__(self, k:int|tuple, s:int|tuple, p:int|tuple, d:int|tuple = 1, maxpool:bool=True):
        super().__init__()
        self._pool = nn.MaxPool2d(k, s, p, d) if maxpool else nn.AvgPool2d(k, s, p, d)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self._pool(x)
    
class AdaptivePool1D(nn.Module):
    def __init__(self, output:int):
        super().__init__()
        self._adaptive_pool = nn.AdaptiveAvgPool1d(output)
        self._flatten = nn.Flatten()
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self._flatten(self._adaptive_pool(x))
class AdaptivePool2D(nn.Module):
    def __init__(self, output:tuple):
        super().__init__()
        self._adaptive_pool = nn.AdaptiveAvgPool2d(output)
        self._flatten = nn.Flatten()
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self._flatten(self._adaptive_pool(x))
    
class MultiHeadPool1D(nn.Module):
    def __init__(self, fine:int=16, balanced:int=4, coarse:int=1):
        super().__init__()
        self._fine_pool = nn.AdaptiveAvgPool1d(fine)
        self._balanced_pool = nn.AdaptiveAvgPool1d(balanced)
        self._coarse_pool = nn.AdaptiveAvgPool1d(coarse)
        self._flatten = nn.Flatten()
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x1 = self._flatten(self._fine_pool(x))
        x2 = self._flatten(self._balanced_pool(x))
        x3 = self._flatten(self._coarse_pool(x))
        return torch.cat([x1, x2, x3], dim=1)
    
class MultiHeadPool2D(nn.Module):
    def __init__(self, fine:tuple=(4,4), balanced:tuple=(2,2), coarse:tuple=(1,1)):
        super().__init__()
        self._fine_pool = nn.AdaptiveAvgPool2d(fine)
        self._balanced_pool = nn.AdaptiveAvgPool2d(balanced)
        self._coarse_pool = nn.AdaptiveAvgPool2d(coarse)
        self._flatten = nn.Flatten()
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x1 = self._flatten(self._fine_pool(x))
        x2 = self._flatten(self._balanced_pool(x))
        x3 = self._flatten(self._coarse_pool(x))
        return torch.cat([x1, x2, x3], dim=1)
    
class MLP(nn.Module):
    def __init__(self, input: int, output: int, channels: int, dropout: float = 0.0, act: nn.Module | None = None, base: int = 8):
        super().__init__()

        act = act if act else nn.Identity()

        dims = [input * (output / input) ** (i / channels) for i in range(channels + 1)]
        dims = [max(base, int(round(d / base)) * base) for d in dims]
        dims[0] = input
        dims[-1] = output

        layers: list[nn.Module] = []
        for i in range(channels):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < channels - 1:
                layers.append(act)
                layers.append(nn.Dropout(dropout))
        self.m = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)


def _require_odd(value: int | tuple, owner: str, param: str = "k") -> None:
    values = value if isinstance(value, tuple) else (value,)
    if any(v % 2 == 0 for v in values):
        raise ValueError(f"{owner} requires odd {param} (per-axis for tuples), got {param}={value}")


def _half(k: int | tuple) -> int | tuple:
    return tuple(v // 2 for v in k) if isinstance(k, tuple) else k // 2


def _is_unit_stride(s: int | tuple) -> bool:
    return all(v == 1 for v in s) if isinstance(s, tuple) else s == 1


class _ResUnit1D(nn.Module):
    def __init__(self, input:int, output:int, k:int, s:int, act:nn.Module):
        super().__init__()
        p = _half(k)
        self._conv1 = Conv1D(input, output, k, s, p, act=act, bn=True)
        self._conv2 = Conv1D(output, output, k, 1, p, act=nn.Identity(), bn=True)
        self._act = act
        if input == output and _is_unit_stride(s):
            self._shortcut = nn.Identity()
        else:
            self._shortcut = Conv1D(input, output, 1, s, 0, act=nn.Identity(), bn=True)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self._act(self._conv2(self._conv1(x)) + self._shortcut(x))


class ResBlock1D(nn.Module):
    """Post-activation ResNet-v1 style residual block.

    With p=k//2 and odd k, a stride-s conv's output length is
    floor((L-1)/s)+1 -- identical to the stride-s 1x1 shortcut conv's
    output length -- so the main path and shortcut are always
    shape-compatible for addition, at any stride. BatchNorm is always
    enabled inside residual blocks (unlike Conv1D's bn=False default)
    because residual stacks train unstably without normalizing the
    pre-addition branches.
    """
    def __init__(self, input:int, output:int, k:int=3, s:int=1, blocks:int=1, act:nn.Module=nn.ReLU()):
        super().__init__()
        _require_odd(k, "ResBlock1D")
        self._blocks = nn.ModuleList([_ResUnit1D(input, output, k, s, act)])
        for _ in range(blocks - 1):
            self._blocks.append(_ResUnit1D(output, output, k, 1, act))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        for block in self._blocks:
            x = block(x)
        return x


class _ResUnit2D(nn.Module):
    def __init__(self, input:int, output:int, k:int|tuple, s:int|tuple, act:nn.Module):
        super().__init__()
        p = _half(k)
        self._conv1 = Conv2D(input, output, k, s, p, act=act, bn=True)
        self._conv2 = Conv2D(output, output, k, 1, p, act=nn.Identity(), bn=True)
        self._act = act
        if input == output and _is_unit_stride(s):
            self._shortcut = nn.Identity()
        else:
            self._shortcut = Conv2D(input, output, 1, s, 0, act=nn.Identity(), bn=True)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self._act(self._conv2(self._conv1(x)) + self._shortcut(x))


class ResBlock2D(nn.Module):
    """Post-activation ResNet-v1 style residual block (2D).

    Same shape-safety identity as ResBlock1D, applied per spatial axis
    independently: with p=k//2 and odd k (checked per axis for tuples),
    the main path and the strided 1x1 shortcut produce identical H and
    W for any stride, since nn.Conv2d computes each output dimension
    with the same 1D formula independently. BatchNorm is always
    enabled inside residual blocks (unlike Conv2D's bn=False default)
    because residual stacks train unstably without normalizing the
    pre-addition branches.
    """
    def __init__(self, input:int, output:int, k:int|tuple=3, s:int|tuple=1, blocks:int=1, act:nn.Module=nn.ReLU()):
        super().__init__()
        _require_odd(k, "ResBlock2D")
        self._blocks = nn.ModuleList([_ResUnit2D(input, output, k, s, act)])
        for _ in range(blocks - 1):
            self._blocks.append(_ResUnit2D(output, output, k, 1, act))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        for block in self._blocks:
            x = block(x)
        return x


class SE1D(nn.Module):
    def __init__(self, channels:int, r:int=2):
        super().__init__()
        hidden = max(1, channels // r)
        self._pool = nn.AdaptiveAvgPool1d(1)
        self._fc1 = nn.Linear(channels, hidden)
        self._act = nn.ReLU()
        self._fc2 = nn.Linear(hidden, channels)
        self._gate = nn.Sigmoid()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        b, c, _ = x.shape
        s = self._gate(self._fc2(self._act(self._fc1(self._pool(x).view(b, c)))))
        return x * s.view(b, c, 1)


class SE2D(nn.Module):
    def __init__(self, channels:int, r:int=2):
        super().__init__()
        hidden = max(1, channels // r)
        self._pool = nn.AdaptiveAvgPool2d(1)
        self._fc1 = nn.Linear(channels, hidden)
        self._act = nn.ReLU()
        self._fc2 = nn.Linear(hidden, channels)
        self._gate = nn.Sigmoid()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        b, c, _, _ = x.shape
        s = self._gate(self._fc2(self._act(self._fc1(self._pool(x).view(b, c)))))
        return x * s.view(b, c, 1, 1)


class ECA1D(nn.Module):
    def __init__(self, channels:int, k:int=3):
        super().__init__()
        _require_odd(k, "ECA1D")
        self._pool = nn.AdaptiveAvgPool1d(1)
        self._conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
        self._gate = nn.Sigmoid()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        b, c, _ = x.shape
        s = self._gate(self._conv(self._pool(x).view(b, 1, c))).view(b, c, 1)
        return x * s


class ECA2D(nn.Module):
    def __init__(self, channels:int, k:int=3):
        super().__init__()
        _require_odd(k, "ECA2D")
        self._pool = nn.AdaptiveAvgPool2d(1)
        self._conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
        self._gate = nn.Sigmoid()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        b, c, _, _ = x.shape
        s = self._gate(self._conv(self._pool(x).view(b, 1, c))).view(b, c, 1, 1)
        return x * s


class CBAM1D(nn.Module):
    def __init__(self, channels:int, r:int=2, spatial_k:int=7):
        super().__init__()
        _require_odd(spatial_k, "CBAM1D", "spatial_k")
        hidden = max(1, channels // r)
        self._avg_pool = nn.AdaptiveAvgPool1d(1)
        self._max_pool = nn.AdaptiveMaxPool1d(1)
        self._fc1 = nn.Linear(channels, hidden)
        self._act = nn.ReLU()
        self._fc2 = nn.Linear(hidden, channels)
        self._channel_gate = nn.Sigmoid()
        self._spatial_conv = nn.Conv1d(2, 1, spatial_k, padding=spatial_k // 2, bias=False)
        self._spatial_gate = nn.Sigmoid()

    def _channel_mlp(self, x:torch.Tensor)->torch.Tensor:
        return self._fc2(self._act(self._fc1(x)))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        b, c, _ = x.shape
        avg = self._channel_mlp(self._avg_pool(x).view(b, c))
        mx = self._channel_mlp(self._max_pool(x).view(b, c))
        x = x * self._channel_gate(avg + mx).view(b, c, 1)
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        s = self._spatial_gate(self._spatial_conv(torch.cat([avg_map, max_map], dim=1)))
        return x * s


class CBAM2D(nn.Module):
    def __init__(self, channels:int, r:int=2, spatial_k:int=7):
        super().__init__()
        _require_odd(spatial_k, "CBAM2D", "spatial_k")
        hidden = max(1, channels // r)
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._max_pool = nn.AdaptiveMaxPool2d(1)
        self._fc1 = nn.Linear(channels, hidden)
        self._act = nn.ReLU()
        self._fc2 = nn.Linear(hidden, channels)
        self._channel_gate = nn.Sigmoid()
        self._spatial_conv = nn.Conv2d(2, 1, spatial_k, padding=spatial_k // 2, bias=False)
        self._spatial_gate = nn.Sigmoid()

    def _channel_mlp(self, x:torch.Tensor)->torch.Tensor:
        return self._fc2(self._act(self._fc1(x)))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        b, c, _, _ = x.shape
        avg = self._channel_mlp(self._avg_pool(x).view(b, c))
        mx = self._channel_mlp(self._max_pool(x).view(b, c))
        x = x * self._channel_gate(avg + mx).view(b, c, 1, 1)
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        s = self._spatial_gate(self._spatial_conv(torch.cat([avg_map, max_map], dim=1)))
        return x * s




if __name__ == '__main__':
    if False:
        x = torch.randn((128, 1, 139, 38))
        first_layer = Conv2D(1, 2, (3,1), 1, (3,0))
        first_pool = Pool2D((2,1), (2,1), (1,0))
        second_layer = Conv2D(2, 4, (3,1), 1, (2,0))
        second_pool = Pool2D((2,1), (2,1), (1,0))
        third_layer = Conv2D(4, 8, 3, 2, 0)
        third_pool = Pool2D(2, 2, 0)
        adaptive_avg_pool = AdaptivePool2D((4,4))
        x = first_layer(x)
        print(x.shape)
        x = first_pool(x)
        print(x.shape)
        x = second_layer(x)
        print(x.shape)
        x = second_pool(x)
        print(x.shape)
        x = third_layer(x)
        print(x.shape)
        x = third_pool(x)
        print(x.shape)
        x = adaptive_avg_pool(x)
        print(x.shape)
    x = torch.randn((1040, 1, 600))
    first_layer = Conv1D(1,2,7,1,0)
    first_pool = Pool1D(2,2,1)
    second_layer = Conv1D(2,4,5,1,0)
    second_pool = Pool1D(2,2,1)
    third_layer = Conv1D(4,8,3,1,0)
    third_pool = Pool1D(2,2,1)
    adaptive_avg_pool = AdaptivePool1D(1)
    x = first_layer(x)
    print(x.shape)
    x = first_pool(x)
    print(x.shape)
    x = second_layer(x)
    print(x.shape)
    x = second_pool(x)
    print(x.shape)
    x = third_layer(x)
    print(x.shape)
    x = third_pool(x)
    print(x.shape)
    x = adaptive_avg_pool(x)
    print(x.shape)
    mlp = MLP(8, 4, 1, 0.1, nn.ReLU())
    x = mlp(x)
    print(x.shape)
    for module in mlp.modules:
        print(module)
 
