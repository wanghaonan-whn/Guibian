import numpy as np
import torch
import collections
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable

EPS = 1e-8
high_size = 2048
low_size = 128
RANDOM_RESOLUTIONS = [512, 768, 1024, 1280, 1536]


class BatchTestResolution(object):
    def __init__(self, size: int = None, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2) or (size is None)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs: np.ndarray):
        h, w = imgs[0].size
        if h > self.size and w > self.size:
            return [transforms.Resize(self.size, self.interpolation)(img) for img in imgs]
        else:
            return imgs


class BatchToTensor(object):
    def __call__(self, imgs):
        return [transforms.ToTensor()(img) for img in imgs]


class BatchRGBToYCbCr(object):
    def __call__(self, imgs):
        return [torch.stack((0. / 256. + img[0, :, :] * 0.299000 + img[1, :, :] * 0.587000 + img[2, :, :] * 0.114000,
                             128. / 256. - img[0, :, :] * 0.168736 - img[1, :, :] * 0.331264 + img[2, :, :] * 0.500000,
                             128. / 256. + img[0, :, :] * 0.500000 - img[1, :, :] * 0.418688 - img[2, :, :] * 0.081312),
                            dim=0) for img in imgs]


class BatchRandomResolution(object):
    def __init__(self, size: int = None, interpolation=Image.BILINEAR):

        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2) or (size is None)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        if self.size is None:
            h, w = imgs[0].size
            max_idx = 0
            for i in range(len(RANDOM_RESOLUTIONS)):
                if h > RANDOM_RESOLUTIONS[i] and w > RANDOM_RESOLUTIONS[i]:
                    max_idx += 1
            idx = np.random.randint(max_idx)
            self.size = RANDOM_RESOLUTIONS[idx]
        return [transforms.Resize(self.size, self.interpolation)(img) for img in imgs]


test_hr_transform = transforms.Compose([
    BatchTestResolution(high_size, interpolation=2),
    BatchToTensor(),
    BatchRGBToYCbCr()
])

test_lr_transform = transforms.Compose([
    BatchRandomResolution(low_size, interpolation=2),
    BatchToTensor(),
    BatchRGBToYCbCr()
])


class YCbCrToRGB(object):
    def __call__(self, img):
        return torch.stack(
            [img[:, 0, :, :] + (img[:, 2, :, :] - 128 / 256.) * 1.402,
             img[:, 0, :, :] - (img[:, 1, :, :] - 128 / 256.) * 0.344136 -
             (img[:, 2, :, :] - 128 / 256.) * 0.714136, img[:, 0, :, :] +
             (img[:, 1, :, :] - 128 / 256.) * 1.772], dim=1)


def preprocess_mefnet(image):
    """ process image for mefnet format"""

    high, low = split_exposure(image)  # HxW

    image_seq = []
    image_seq.append(Image.fromarray(high).convert('RGB'))
    image_seq.append(Image.fromarray(low).convert('RGB'))

    I_he = test_hr_transform(image_seq)
    I_le = test_hr_transform(image_seq)

    I_he = torch.stack(I_he, 0).contiguous()
    I_le = torch.stack(I_le, 0).contiguous()
    i_he = I_he.cuda(non_blocking=True)
    i_le = I_le.cuda(non_blocking=True)
    i_he = torch.squeeze(i_he, dim=0)
    i_le = torch.squeeze(i_le, dim=0)

    Y_he = i_he[:, 0, :, :].unsqueeze(1)  # 高曝图片的亮度
    Cb_he = i_he[:, 1, :, :].unsqueeze(1)
    Cr_he = i_he[:, 2, :, :].unsqueeze(1)

    Wb = (torch.abs(Cb_he - 0.5) + EPS) / torch.sum(torch.abs(Cb_he - 0.5) + EPS, dim=0)
    Wr = (torch.abs(Cr_he - 0.5) + EPS) / torch.sum(torch.abs(Cr_he - 0.5) + EPS, dim=0)
    Cb_f = torch.sum(Wb * Cb_he, dim=0, keepdim=True).clamp(0, 1)
    Cr_f = torch.sum(Wr * Cr_he, dim=0, keepdim=True).clamp(0, 1)

    Y_le = i_le[:, 0, :, :].unsqueeze(1)
    I_he = Variable(Y_he)
    I_le = Variable(Y_le)
    return I_he, I_le, Cb_f, Cr_f


def split_exposure(image: np.ndarray):
    """ 线阵相机高低曝光拆分 """
    low = image[0::2]
    high = image[1::2]

    return high, low


def tensor_to_uint8_hwc(img_chw_float_0_1: torch.Tensor) -> np.ndarray:
    """
    img_chw_float_0_1: CPU 上的 [3,H,W] float tensor, 值域 0~1
    return: HWC uint8 ndarray
    """
    img = img_chw_float_0_1.clamp(0, 1).mul(255).byte()
    return img.permute(1, 2, 0).contiguous().numpy()
