# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import REPRESENTORS, build_backbone, build_learnable_params
from ..heads import MultiLabelClsHead
from ..utils.augment import Augments
from ..classifiers.base import BaseClassifier
from mmcv.runner import BaseModule, auto_fp16


@REPRESENTORS.register_module()
class ImageRepresentor(BaseClassifier):

    def __init__(self,
                 backbone,
                 modulations,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(ImageRepresentor, self).__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        
        self.siren = build_backbone(backbone)
        self.modulations = build_learnable_params(modulations)

    def extract_feat(self, img, stage='backbone'):
        x = self.backbone(img)

        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.neck(x)
        if stage == 'neck':
            return x

        if self.with_head and hasattr(self.head, 'pre_logits'):
            x = self.head.pre_logits(x)
        return x
    
    @auto_fp16(apply_to=('img', ))
    def forward(self, img, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)

    def forward_train(self, img, **kwargs):
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        x = self.extract_feat(img)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)

        losses.update(loss)

        return losses

    def simple_test(self, img, img_metas=None, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)

        if isinstance(self.head, MultiLabelClsHead):
            assert 'softmax' not in kwargs, (
                'Please use `sigmoid` instead of `softmax` '
                'in multi-label tasks.')
        res = self.head.simple_test(x, **kwargs)

        return res
