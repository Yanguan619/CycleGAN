import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import Module
from torchvision import transforms

from unit.trainer import cycle_gan, munit
from unit import models
from unit.datasets.one_dataset import OneDataset
from unit.datasets.single_dataset import SingleDataset
from unit.utils import show_img, tensor2np, show_imgs


def load_model(model, style: str = None):
    if isinstance(model, models.CycleGANModel):
        state_dict_path = f"weights/cycle_gan/{style}"
        if style.split(".")[-1] == "pth":
            cycle_gan.load(model.net_G_A, state_dict_path)
        else:
            state_dict = torch.load(state_dict_path)["state_dict"]
            # strict=False,只匹配存在的参数
            model.load_state_dict(state_dict, strict=False)
    elif isinstance(model, models.MUNITModel):
        state_dict_path = f"weights/munit/{style}"
        munit.load(model, state_dict_path)
    elif isinstance(model, models.FUNITModel):
        state_dict_path = f"weights/funit/{style}"
        state_dict = torch.load(state_dict_path)
        model.net_G.load_state_dict(state_dict["gen"])
    else:
        state_dict_path = None
    print(f"Loaded {state_dict_path}")


def detect(
    image: str = None,
    styles: str | list = None,
    show_image=False,
    save_image=False,
    model=None,
    class_img_folder_dir: str | list = "unit/inputs/meerkat",
):
    print("🔥 Prepare detect")
    results = []
    if image is None:
        image = "unit/inputs/horse.jpg"
    if styles is None:
        styles = "horse2zebra.pth"
    if model is None:
        model = models.CycleGANModel(mode="test")

    if isinstance(styles, str):
        styles = [styles]
    for style in styles:
        load_model(model, style)
        if isinstance(image, str):
            if Path(image).is_file():
                img = SingleDataset(image)[0]["A"]
            elif Path(image).is_dir():
                img = OneDataset(image, None)
            else:
                assert "image is not file and dir"
        elif image is None:
            assert "img is None"
        else:
            img = OneDataset(image, None)[0]

        if isinstance(model, models.FUNITModel):
            # FUNITModel 模型需要将图片裁剪为128并得到目标图像编码，故单独重新读取
            transform = transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            if isinstance(image, str):
                img = Image.open(image).convert("RGB")
                img = transform(img)
            else:
                img = transform(image)
            if (
                isinstance(class_img_folder_dir, str)
                and Path(class_img_folder_dir).is_dir()
            ):
                imgs = [
                    Path(class_img_folder_dir, i)
                    for i in os.listdir(class_img_folder_dir)
                ]
            else:
                imgs = class_img_folder_dir
            for i, class_img_path in enumerate(imgs):
                class_img = transform(Image.open(class_img_path).convert("RGB"))
                with torch.no_grad():
                    class_code = model.compute_k_style(class_img, 1)
                    if i == 0:
                        new_class_code = class_code
                    else:
                        new_class_code += class_code
            final_class_code = new_class_code / len(imgs)

        print(f"Detect [{style}]")
        model.eval()
        with torch.no_grad():
            if isinstance(model, models.FUNITModel):
                output = model(img, final_class_code).cpu().detach()
            else:
                output = model(img).cpu().detach()
        results.append(tensor2np(output))

    if show_image or save_image:
        for result in results:
            if show_image:
                show_imgs([tensor2np(img), result])
                import ssim
                from unit.utils.scorce import calc_ssim, calc_psnr, SegmentationMetric

                # ssim_value = ssim.ssim(img, output).item()
                ssim_value = calc_ssim(tensor2np(img), result)
                ssim_value = calc_ssim(tensor2np(img), result)
                psnr_value = calc_psnr(tensor2np(img), result)

                print("Initial ssim:", ssim_value)
                print("Initial psnr:", psnr_value)

            if save_image:
                show_img(result, save_image=save_image)
    else:
        return results
