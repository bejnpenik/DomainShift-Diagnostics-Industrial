import torch
import torch.nn as nn
from model.modules import *

class Encoder1D(nn.Module):
    def __init__(self, input_channels:int=1, dropout_level:float=0.1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            Conv1D(input_channels, 2, 7, 1, 0),
            Pool1D(2, 2, 1),
            nn.Dropout(dropout_level),
            Conv1D(2, 4, 5, 1, 0),
            Pool1D(2, 2, 1),
            nn.Dropout(dropout_level),
            Conv1D(4, 8, 3, 1, 0),
            Pool1D(2, 2, 1)
        )
        self.output_channels:int = 8
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.conv_layers(x)
        
class Encoder2D(nn.Module):
    def __init__(self, input_channels:int=1, dropout_level:float=0.1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            Conv2D(input_channels, 2, (3,1), 1, (3,0)),
            Pool2D((2,1), (2,1), (1,0)),
            nn.Dropout2d(dropout_level),
            Conv2D(2, 4, (3,1), 1, (2,0)),
            Pool2D((2,1), (2,1), (1,0)),
            nn.Dropout2d(dropout_level),
            Conv2D(4, 8, 3, 2, 0),
            Pool2D(2, 2, 0)
        )
        self.output_channels:int = 8
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.conv_layers(x)
    
class FeatureAggregator1D(nn.Module):
    def __init__(self, output_features:int|tuple=16, multi_head:bool=False):
        # if tuple is supplied and mh = false raise error 
        # for sanity check that you know what you are doing
        super().__init__()
        if isinstance(output_features, tuple) and not multi_head:
            raise ValueError(f'output_features is represented as tuple however multi_head was not checked. quitting')
        if isinstance(output_features, tuple):
            if len(output_features) != 3:
                raise ValueError(f'output_features should be length of 3.')
            fine, balanced, coarse = output_features
            if fine > balanced and balanced > coarse:
                self.adaptive_pooling = MultiHeadPool1D(fine, balanced, coarse)
                self.output_features:int = fine + balanced + coarse
            else:
                raise ValueError(f'Progresions should be fine > balanced > coarse for the entries of output_features tuple.')
        else:
            self.adaptive_pooling = AdaptivePool1D(output_features)
            self.output_features:int = output_features
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.adaptive_pooling(x)
    
class FeatureAggregator2D(nn.Module):
    def __init__(self, output_features:int|tuple=16, multi_head:bool=False):
        # if tuple is supplied and mh = false raise error 
        # for sanity check that you know what you are doing
        super().__init__()
        if isinstance(output_features, tuple) and not multi_head:
            raise ValueError(f'output_features is represented as tuple however multi_head was not checked. quitting')
        if isinstance(output_features, tuple):
            if len(output_features) != 3:
                raise ValueError(f'output_features should be length of 3.')
            fine, balanced, coarse = output_features
            if fine > balanced and balanced > coarse:
                from math import sqrt
                fine_sqrt = int(sqrt(fine)) # rounding idc
                balanced_sqrt = int(sqrt(balanced)) # rounding idc
                coarse_sqrt = int(sqrt(coarse)) # rounding idc
                self.adaptive_pooling = MultiHeadPool2D(
                    (fine_sqrt, fine_sqrt), 
                    (balanced_sqrt, balanced_sqrt), 
                    (coarse_sqrt, coarse_sqrt)
                    )
                self.output_features:int = fine_sqrt * fine_sqrt 
                self.output_features += balanced_sqrt * balanced_sqrt 
                self.output_features += coarse_sqrt * coarse_sqrt 
            else:
                raise ValueError(f'Progresions should be fine > balanced > coarse for the entries of output_features tuple.')
        else:
            from math import sqrt
            output_features_sqrt = int(sqrt(output_features)) # rounding idc
            self.adaptive_pooling = AdaptivePool2D((output_features_sqrt, output_features_sqrt))
            self.output_features:int = output_features_sqrt * output_features_sqrt
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.adaptive_pooling(x)
    
class ClassificationHead(nn.Module):
    def __init__(self, input_features:int, num_classes:int=4, depth:int=2, dropout_level:float=0.1, act:nn.Module|None=nn.ReLU()):
        super().__init__()
        self.fc_layers = MLP(input_features, num_classes, depth, dropout_level, act)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.fc_layers(x)
    

class CNN1D(nn.Module):
    def __init__(self, 
                input_channels:int=1,
                num_classes:int=4,
                aggregator_levels:int|tuple=1,
                head_depth:int=2,
                head_act:nn.Module|None=nn.ReLU(),
                dropout_levels:tuple=(0.1, 0.1)):
        super().__init__()
        self.encoder = Encoder1D(input_channels, dropout_levels[0])
        self.aggregator = FeatureAggregator1D(aggregator_levels, isinstance(aggregator_levels, tuple))
        self.head = ClassificationHead(
            self.aggregator.output_features * self.encoder.output_channels,
            num_classes,
            head_depth,
            dropout_levels[1],
            head_act,
        )
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.head(self.aggregator(self.encoder(x)))
class CNN2D(nn.Module):
    def __init__(self, 
                input_channels:int=1,
                num_classes:int=4,
                aggregator_levels:int|tuple=1,
                head_depth:int=2,
                head_act:nn.Module|None=nn.ReLU(),
                dropout_levels:tuple=(0.1, 0.1)):
        super().__init__()
        self.encoder = Encoder2D(input_channels, dropout_levels[0])
        self.aggregator = FeatureAggregator2D(aggregator_levels, isinstance(aggregator_levels, tuple))
        self.head = ClassificationHead(
            self.aggregator.output_features * self.encoder.output_channels,
            num_classes,
            head_depth,
            dropout_levels[1],
            head_act
        )
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.head(self.aggregator(self.encoder(x)))