import numpy as np
import torch
import torch.nn as nn


def cpt_ssim(img, img_gt, normalize=False):

    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        img_gt = (img_gt - img_gt.min()) / (img_gt.max() - img_gt.min() + 1e-9)

    SSIM = sk_cpt_ssim(img, img_gt, data_range=img_gt.max() - img_gt.min())

    return SSIM


def cpt_psnr(img, img_gt, PIXEL_MAX=1.0, normalize=False):

    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        img_gt = (img_gt - img_gt.min()) / (img_gt.max() - img_gt.min() + 1e-9)

    mse = np.mean((img - img_gt) ** 2)
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    return psnr


def cpt_cos_similarity(img, img_gt, normalize=False):

    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        img_gt = (img_gt - img_gt.min()) / (img_gt.max() - img_gt.min() + 1e-9)

    cos_dist = np.sum(img*img_gt) / np.sqrt(np.sum(img**2)*np.sum(img_gt**2) + 1e-9)

    return cos_dist


def cpt_batch_psnr(img, img_gt, PIXEL_MAX):
    mse = torch.mean((img - img_gt) ** 2)
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return psnr


def psnr(pred, target, max_pixel=1.0, normalize=False):

    assert isinstance(pred, (torch.Tensor, np.ndarray)), \
        f'The pred should be torch.Tensor or np.ndarray ' \
        f'instead of {type(pred)}.'
    assert isinstance(target, (torch.Tensor, np.ndarray)), \
        f'The target should be torch.Tensor or np.ndarray ' \
        f'instead of {type(target)}.'

    # torch version is faster in most situations.
    to_tensor = (lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)

    pred = to_tensor(pred)
    target = to_tensor(target)
    if normalize:
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-9)
        target = (target - target.min()) / (target.max() - target.min() + 1e-9)

    mse = torch.mean((pred - target) ** 2)
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))

    return psnr


class PSNR(nn.Module):
    def __init__(self, max_pixel=1.0, normalize=False):
        super().__init__()
        self.max_pixel = max_pixel
        self.normalize = normalize

    def forward(self, pred, target):
        return psnr(pred, target, self.max_pixel, self.normalize)
