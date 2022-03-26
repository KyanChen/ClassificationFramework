# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import PARAMETERS, build_learnable_params
from ..heads import MultiLabelClsHead
from ..utils.augment import Augments
from ..classifiers.base import BaseClassifier
from mmcv.runner import BaseModule, auto_fp16

import torch
from torch import nn as nn


@PARAMETERS.register_module()
class Modulations(BaseModule):
    def __init__(self,
                 n_dims,
                 n_batch,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(Modulations, self).__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.modulations = nn.Parameter(torch.zeros(n_batch, n_dims))
    
    def forward(self):
        return self.modulations
    
    def freeze_model(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def set_zeros(self):
        self.modulations.data = torch.zeros_like(self.modulations)

    def train(self, mode: bool = True):
        super(Modulations, self).train(mode)
        for param in self.parameters():
            param.requires_grad = True
