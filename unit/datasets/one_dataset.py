from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from .base_dataset import get_transform
from torchvision import transforms


class OneDataset(Dataset):
    """加载数据

    加载文件夹中所有图片或直接加载指定文件, dataroot 传入的直接是PIL格式图片
    """

    def __init__(self, img, opt):
        self.opt = opt
        self.A_img = [img]
        # self.transform = get_transform(opt, grayscale=(opt.input_nc == 1))
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __getitem__(self, idx: int):
        A_img = self.A_img[idx]
        A = self.transform(A_img)
        return  A

    def __len__(self):
        return 1
