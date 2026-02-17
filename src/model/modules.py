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
    def __init__(self, input:int, output:int, channels:int, dropout:float=0.0, act:nn.Module|None=None, base = 8):
        super().__init__()

        self._dropout = nn.Dropout(dropout)

        self._act = act if act else nn.Identity()

        self._dims = [
            input * (output / input) ** (i / channels)
            for i in range(channels + 1)
        ]

        self._dims = [max(base, int(round(d / base)) * base) for d in self._dims]
        self._dims[0] = input
        self._dims[-1] = output
        self.layers = []
        for i in range(channels):
            self.layers.append(
                nn.Linear(self._dims[i], self._dims[i+1])
            )
            if i < channels - 1:
                self.layers.append(self._act)
                self.layers.append(self._dropout)
        self.m = nn.Sequential(*self.layers)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self.m(x)
    




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
 
