

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