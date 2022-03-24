from mmcv.runner import HOOKS, Hook
import mmcv


@HOOKS.register_module()
class VisImg(Hook):
    def __init__(self, dir, **kwargs):
        self.dir = dir
        mmcv.mkdir_or_exist(dir)


    def vis_batch_img(self, pred, gt, name, img_metas):




