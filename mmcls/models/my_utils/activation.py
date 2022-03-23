import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import Registry
from mmcv.cnn.bricks import ACTIVATION_LAYERS


from mmcv.utils import TORCH_VERSION, build_from_cfg, digit_version



@ACTIVATION_LAYERS.register_module()
class Sine(nn.Module):
    def __init__(self, w0=30.):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x, modulations):
        x = torch.sin(self.w0 * (x + modulations))
        return x

