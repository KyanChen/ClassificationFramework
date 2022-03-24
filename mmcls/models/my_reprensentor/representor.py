# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import REPRESENTORS, build_backbone, build_learnable_params, build_loss
from ..heads import MultiLabelClsHead
from ..utils.augment import Augments
from ..classifiers.base import BaseClassifier
from mmcv.runner import BaseModule, auto_fp16
from einops import rearrange, repeat
from collections import OrderedDict
import torch.distributed as dist
from mmcls.models.losses import PSNR



@REPRESENTORS.register_module()
class ImageRepresentor(BaseClassifier):

    def __init__(self,
                 backbone,
                 modulations,
                 pretrained=None,
                 img_size=None,
                 max_sample_per_img=0.3,
                 max_inner_iter=10,
                 cal_acc=True,
                 loss=dict(type='SmoothL1Loss', loss_weight=1.0),
                 train_cfg=None,
                 init_cfg=None):
        super(ImageRepresentor, self).__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        
        self.siren = build_backbone(backbone)
        self.modulations = build_learnable_params(modulations)
        w, h = img_size
        self.grid = self.get_grid(h, w)
        if isinstance(max_sample_per_img, float):
            max_sample_per_img = int(h*w*max_sample_per_img)
        self.max_sample_per_img = max_sample_per_img
        self.max_inner_iter = max_inner_iter
        self.compute_loss = build_loss(loss)
        self.compute_accuracy = PSNR()
        self.cal_acc = cal_acc

    def get_grid(self, h, w, is_normalize=True):
        j = torch.linspace(0.0, h - 1.0, h)
        i = torch.linspace(0.0, w - 1.0, w)
        if is_normalize:
            j = j / (h-1) - 0.5
            i = i / (w-1) - 0.5
        j_grid, i_grid = torch.meshgrid(j, i)
        grid = 2 * torch.stack([i_grid, j_grid])

        # tensors = [torch.linspace(-1, 1, steps = image_height), torch.linspace(-1, 1, steps = image_width)]
        # mgrid = torch.stack(torch.meshgrid(*tensors, indexing = 'ij'), dim=-1)
        # mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        return grid  # C H W

    def extract_feat(self, imgs, stage=None):
        pass

    # @auto_fp16(apply_to=('img', ))
    def forward(self, img, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)

    def forward_train(self, img, **kwargs):
        B, C, H, W = img.size()

        gt_labels = img
        gt_label_samples = rearrange(gt_labels, 'b c h w -> (b h w) c')  # n_batch, n_sample, 3
        imgs = repeat(self.grid, 'c h w -> B c h w', B=B)
        samples_per_img = H * W
        img_samples = rearrange(imgs, 'b c h w -> (b h w) c')  # n_batch, n_sample, 2

        runner = kwargs['runner']
        # 冻结siren参数
        self.modulations.train(True)
        self.siren._freeze_model()
        # modulations 置零
        self.modulations._set_zeros()

        for i_iter in range(self.max_inner_iter):
            for hook in runner.hooks:
                if hasattr(hook, 'before_train_inner_iter'):
                    getattr(hook, 'before_train_inner_iter')(runner, i_iter)

            # sample
            sample_inds = [torch.randint(0, samples_per_img, size=[self.max_sample_per_img]) + idx * samples_per_img for
                           idx in range(B)]
            sample_inds = torch.cat(sample_inds)
            inputs = img_samples[sample_inds]
            targets = gt_label_samples[sample_inds]
            modulations = repeat(self.modulations.modulations, 'b n_dims -> (b n_samples) n_dims',
                                 n_samples=self.max_sample_per_img)
            x = self.siren(inputs, modulations)
            num_samples = len(x)
            loss = self.compute_loss(x, targets, avg_factor=num_samples)
            # 求梯度
            for hook in runner.hooks:
                if hasattr(hook, 'after_train_inner_iter'):
                    getattr(hook, 'after_train_inner_iter')(runner, 'modulations', loss)

        self.modulations._freeze_model()
        self.siren.train(True)

        inputs = img_samples
        targets = gt_label_samples
        modulations = repeat(self.modulations.modulations, 'b n_dims -> (b n_samples) n_dims',
                             n_samples=samples_per_img)
        x = self.siren(inputs, modulations)
        num_samples = len(x)
        loss = self.compute_loss(x, targets, avg_factor=num_samples)

        losses = dict()
        losses['loss'] = loss
        if self.cal_acc:
            acc = self.compute_accuracy(x, targets)
            losses['psnr'] = acc

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

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, **kwargs):
        losses = self(**data, **kwargs)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    def val_step(self, data, **kwargs):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs
