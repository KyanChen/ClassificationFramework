from collections import OrderedDict
import warnings
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init)
from mmcv.cnn.bricks import DropPath, build_activation_layer
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


@BACKBONES.register_module()
class ResMLP(BaseBackbone):
    def __init__(self,
                 num_layers,
                 in_channels=2,
                 out_channels=None,
                 base_channels=None,
                 bias=True,
                 expansions=[1],
                 init_cfg=None,
                 ):
        super(ResMLP, self).__init__(init_cfg)
        if len(expansions) == 1:
            self.expansions = expansions * num_layers
        assert num_layers == len(self.expansions)
        out_channels_list = [x * base_channels for x in self.expansions]

        self.layers = []
        layer_0 = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=bias),
            nn.BatchNorm1d(in_channels),
            # nn.GELU(),
            nn.Linear(in_channels, out_channels_list[0], bias=bias),
            nn.BatchNorm1d(out_channels_list[0]),
            # nn.GELU()
        )

        self.add_module('layer_0', layer_0)
        self.layers.append('layer_0')
        self.num_layers = num_layers
        for idx in range(num_layers-1):
            in_channels_ = out_channels_list[idx]
            layer = nn.Sequential(
                nn.Linear(in_channels_, out_channels_list[idx+1]//2, bias=bias),
                nn.BatchNorm1d(out_channels_list[idx + 1] // 2),
                nn.GELU(),
                nn.Dropout(p=0.15),
                nn.Linear(out_channels_list[idx+1]//2, out_channels_list[idx + 1], bias=bias),
                nn.BatchNorm1d(out_channels_list[idx + 1]),
                nn.GELU(),
                nn.Dropout(p=0.15),

            )
            self.add_module(f'layer_inner_{idx}', layer)
            self.layers.append(f'layer_inner_{idx}')

        layer = nn.Sequential(
            nn.Linear(out_channels_list[-1], out_channels_list[-1]//2, bias=bias),
            nn.BatchNorm1d(out_channels_list[-1] // 2),
            nn.GELU(),

            nn.Linear(out_channels_list[-1]//2, out_channels, bias=bias)
        )
        self.add_module(f'layer_last', layer)
        self.layers.append(f'layer_last')

    def forward(self, x):
        for layer_name in self.layers:
            layer = getattr(self, layer_name)
            if 'layer_inner' in layer_name:
                x = layer(x) + x
                # x = layer(x)
            else:
                x = layer(x)
        return x