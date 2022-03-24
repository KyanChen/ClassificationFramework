# from ..builder import BACKBONES
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
eps = 1.0e-5


class SirenLayer(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_fcs=1,
                 bias=True,
                 act_cfg=dict(type='Sine', w0=30.),
                 init_cfg=dict(type='Uniform', a=0, b=1),
                 **kwargs):
        super(SirenLayer, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_cfg = act_cfg
        if act_cfg is None:
            self.activation_func = nn.Identity()
        else:
            self.activation_func = build_activation_layer(act_cfg)

        _in_channels = in_channels
        _out_channels = out_channels
        self.layers = []
        for i in range(0, num_fcs):
            self.add_module(f'layer_{i}_fc', nn.Linear(_in_channels, _out_channels, bias=bias))
            self.add_module(f'layer_{i}_actfunc', self.activation_func)
            self.layers.append([
                f'layer_{i}_fc', 
                f'layer_{i}_actfunc'
                ])
            _in_channels = _out_channels

    def init_weights(self):
        super(SirenLayer, self).init_weights()
    
    def forward(self, x, *args):
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name[0])
            x = layer(x)
            layer = getattr(self, layer_name[1])
            x = layer(x, *args)
        return x


@BACKBONES.register_module()
class Siren(BaseBackbone):
    def __init__(self,
                 inner_layers,
                 in_channels=2,
                 out_channels=3,
                 base_channels=28,
                 num_modulation=256,
                 bias=True,
                 expansions=[1],
                 init_cfg=None,
                 ):
        
        super(Siren, self).__init__(init_cfg)
        if len(expansions) == 1:
            self.expansions = expansions * inner_layers
        assert inner_layers == len(self.expansions)
        if isinstance(self.expansions, list):
            self.expansions = torch.tensor(self.expansions)

        self.inner_layers = inner_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bias = bias
        
        self.layers = []
        _in_channels = in_channels
        out_channels_list = base_channels * self.expansions
        out_channels_list = torch.cat((out_channels_list, torch.tensor([self.out_channels])))
        for i in range(self.inner_layers+1):
            _out_channels = out_channels_list[i]

            w0 = 30.
            if i == 0:
                w_std = 1. / _in_channels
            else:
                c = 6.
                w_std = torch.sqrt(c / _in_channels) / w0
            init_cfg = dict(type='Uniform', a=-w_std, b=w_std)
            if i == self.inner_layers:
                act_cfg = dict(type='Sigmoid')
            else:
                act_cfg = dict(type='Sine', w0=w0)

            layer = SirenLayer(
                _in_channels, _out_channels, num_fcs=1,
                bias=bias, act_cfg=act_cfg,
                init_cfg=init_cfg
            )
            _in_channels = _out_channels
            layer_name = f'SirenLayer_{i}'
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)
        
        self.modulation_size_dict = self.get_bias_size()
        _out_channels = sum(self.modulation_size_dict.values())
        self.shift_modulation_layer = nn.Linear(num_modulation, _out_channels, bias=bias)

    def get_bias_size(self):
        parameters_size = OrderedDict()
        for name, parm in self.named_parameters():
            if '.weight' in name:
                parameters_size[name.replace('.weight', '.bias')] = parm.size(0)
        parameters_size.popitem(last=True)
        return parameters_size

    def get_parameters_size(self):
        parameters_size = dict()
        for name, parm in self.named_parameters():
            parameters_size[name] = parm.size()
        return parameters_size

    def _freeze_model(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def init_weights(self):
        super(Siren, self).init_weights()

    def forward(self, x, modulations):
        shift_modulations = self.shift_modulation_layer(modulations)
        shift_modulations_split = torch.split(shift_modulations, list(self.modulation_size_dict.values()), dim=1)
        shift_modulations_split = shift_modulations_split + (None, )
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            if shift_modulations_split[i] is not None:
                x = layer(x, shift_modulations_split[i])
            else:
                x = layer(x)
        
        return x

    def train(self, mode=True):
        super(Siren, self).train(mode)
        for param in self.parameters():
            param.requires_grad = True

