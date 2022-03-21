import torch
from collections import OrderedDict

checkpoint = 'pretrain/checkpoint_res50.tar'
checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
new_dict = OrderedDict()

for key, value in checkpoint.items():
    key = key.replace('encoder.', 'backbone.')
    key = key.replace('projector.0', 'head.fc.0')
    key = key.replace('projector.2', 'head.fc.1')
    new_dict[key] = value

torch.save(new_dict, 'pretrain/RL_res50.pth')
