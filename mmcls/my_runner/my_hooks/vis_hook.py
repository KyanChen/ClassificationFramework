import os.path

import numpy as np
from einops import rearrange
from mmcv.runner import HOOKS, Hook
import mmcv
from skimage import io


@HOOKS.register_module()
class VisImg(Hook):
    def __init__(self, dir, **kwargs):
        self.dir = dir
        mmcv.mkdir_or_exist(dir)

    def vis_batch_img(self, pred, gt, img_metas, psnr):
        num_samples = len(pred)
        pred = pred.permute(0, 2, 3, 1).data.cpu().numpy()
        gt = gt.permute(0, 2, 3, 1).data.cpu().numpy()
        for img_i in range(num_samples):
            vis_0 = pred[img_i] * 255
            vis_1 = gt[img_i] * 255
            vis = np.vstack((vis_0, vis_1)).astype(np.uint8)
            filename = os.path.basename(img_metas[img_i]['filename']).split('.')[0]
            io.imsave(self.dir+'/'+filename+f'_{int(psnr*100)}.jpg', vis)





