from os.path import dirname, exists, join
import copy
import torch
import numpy as np
import sys
print(sys.path)
sys.path.append(f'{sys.path[0]}/../..')
print(sys.path)


def _get_detector_cfg(config_fpath):
    from mmcv import Config
    config = Config.fromfile(config_fpath)
    model = copy.deepcopy(config.model)
    return model

def _demo_mm_inputs(input_shape=(2, 100, 5)): 
    # (N, n_sam, C) = input_shape
    N, H, W, C = input_shape


    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    # img_h, img_w, _
    img_metas = [{
        'img_shape': (512, 512, 3),
        'ori_shape': (512, 512, 3),
        'pad_shape': (512, 512, 3),
        'filename': '<demo>.png',
        'scale_factor': np.array([1.1, 1.2, 1.1, 1.2]),
        'flip': False,
        'flip_direction': None,
        'batch_input_seq_len': 100,
        'seq_len': 50
    } for _ in range(N)]

    gt_bboxes = []
    gt_labels = []

    for batch_idx in range(N):
        cx, cy, bw, bh = rng.rand(3, 4).T

        tl_x = (cx - bw / 2).clip(0, 1) * 512
        tl_y = (cy - bh / 2).clip(0, 1) * 512
        br_x = (cx + bw / 2).clip(0, 1) * 512
        br_y = (cy + bh / 2).clip(0, 1) * 512

        boxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
        class_idxs = np.array([1, 1, 3])

        gt_bboxes.append(torch.FloatTensor(boxes))
        gt_labels.append(torch.LongTensor(class_idxs))

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'gt_bboxes_ignore': None
    }

    return mm_inputs


def test_backbone():
    model = dict(
        type='Siren', 
        num_layers=3,
        bias=False
    )
    from mmcls.models import build_backbone
    backbone = build_backbone(model)
    inputs = (torch.randn(5, 2), torch.randn(5, 256))
    backbone(*inputs)
    pass

    

if __name__ == '__main__':
    test_backbone()