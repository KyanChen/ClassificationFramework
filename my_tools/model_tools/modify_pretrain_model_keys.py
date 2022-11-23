import torch
from collections import OrderedDict

checkpoint_path = r'I:\CodeRep\ClassificationFramework\tools\results\EXP20220501_5\latest.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict']
new_dict = OrderedDict()

for key, value in checkpoint.items():
    key = key.replace('backbone.', '')
    # key = 'backbone.' + key
    new_dict[key] = value

torch.save(new_dict, checkpoint_path)
