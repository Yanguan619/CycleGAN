import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == "resize_and_crop":
        new_h = new_w = opt.load_size
    elif opt.preprocess == "scale_width_and_crop":
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {"crop_pos": (x, y), "flip": flip}


def get_transform(
    opt,
    load_size: int = 286,
    crop_size: int = 256,
    grayscale=False,
    params=None,
    convert=True,
    method=transforms.InterpolationMode.BICUBIC,
):
    """数据预处理"""
    transform_list = []

    # 灰度化
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    # 图片大小调整
    if opt:
        # 默认：双三次插值
        if "resize" in opt:
            osize = [load_size, load_size]
            transform_list.append(transforms.Resize(osize, method))
        elif "scale_width" in opt:
            transform_list.append(
                transforms.Lambda(
                    lambda img: __scale_width(img, load_size, crop_size, method)
                )
            )
        # 裁剪
        if "crop" in opt:
            if params is None:
                transform_list.append(transforms.RandomCrop(crop_size))
            else:
                transform_list.append(
                    transforms.Lambda(
                        lambda img: __crop(img, params["crop_pos"], crop_size)
                    )
                )
        if opt == "none":
            transform_list.append(
                transforms.Lambda(
                    lambda img: __make_power_2(img, base=4, method=method)
                )
            )
        if "flip" in opt:
            # 图片左右翻转
            if params is None:
                transform_list.append(transforms.RandomHorizontalFlip())
            elif params["flip"]:
                transform_list.append(
                    transforms.Lambda(lambda img: __flip(img, params["flip"]))
                )
    # convert
    if convert:
        transform_list += [transforms.ToTensor()]
        # transform_list += [GaussionNoise()] if opt.isTrain else []
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __transforms2pil_resize(method):
    mapper = {
        transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
        transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
        transforms.InterpolationMode.NEAREST: Image.NEAREST,
        transforms.InterpolationMode.LANCZOS: Image.LANCZOS,
    }
    return mapper[method]


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    """根据给定的方法（例如：双三次插值），将图片变成指定的大小。
    其中的round函数是一种四舍五入的方法。
    """
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img
    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(
    img, target_size: int, crop_size: int, method=transforms.InterpolationMode.BICUBIC
):
    """调整大小"""
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size: int):
    """图片裁剪"""
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip: bool):
    """图片左右翻转"""
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def _gaussion_noise(img):
    noise = torch.randn(img.shape)
    img = img + noise * 0.1
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, "has_printed"):
        print(
            "The image size needs to be a multiple of 4. "
            "The loaded image size was (%d, %d), so it was adjusted to "
            "(%d, %d). This adjustment will be done to all images "
            "whose sizes are not multiples of 4" % (ow, oh, w, h)
        )
        __print_size_warning.has_printed = True


class GaussionNoise:
    """添加高斯噪声"""

    def __init__(self) -> None:
        pass

    def __call__(self, img):
        noise = torch.randn(img.shape)
        img_mix_noise = img + noise * 0.1
        return img_mix_noise

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
