"""不同的模型使用不同的数据集

比如有监督模型使用的都是成对的训练数据、无监督模型使用的数据集不必使用成对的数据
This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying a flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import os
from pathlib import Path

import torch.utils.data

import torch.utils.data as data

from PIL import Image


def find_dataset_class_by_mode(dataset_name: str):
    """按照数据集类型来寻找对应的模块进行动态导入
    Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset = None
    dataset_module_name = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_module_name)

    target_dataset_name = dataset_name.replace("_", "") + "dataset"  # 模块中的类名
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset = cls
    if dataset is None:
        raise NotImplementedError(
            f"In {dataset_module_name}.py, there should be a subclass of BaseDataset with class "
            f"name that matches {target_dataset_name} in lowercase."
        )
    return dataset


def get_option_setter(dataset_name: str):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_class_by_mode(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threading data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threading data loader.
        """
        self.opt = opt
        # 判断数据集类型（成对/不成对），得到相应的类包
        dataset_class = find_dataset_class_by_mode(opt.dataset_mode)
        # 传入数据集路径到类包中，得到数据集
        self.dataset = dataset_class(opt)
        # dataset_file = f"datasets/{opt.name}.pkl"
        # if not Path(dataset_file).exists():
        #     # 判断数据集类型（成对/不成对），得到相应的类包
        #     dataset_class = find_dataset_class_by_mode(opt.dataset_mode)
        #     # 传入数据集路径到类包中，得到数据集
        #     self.dataset = dataset_class(opt)
        #     # 打包下次直接使用
        #     # 打包后文件也很大，暂时就这样
        #     print("pickle dump dataset...")
        #     pickle.dump(self.dataset, open(dataset_file, 'wb'))
        # else:
        #     print("pickle load dataset...")
        #     self.dataset = pickle.load(open(dataset_file, 'rb'))
        # print("dataset [%s] was created" % type(self.dataset).__name__)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
        )

    def load_data(self):
        print(f"The number of training images = {len(self)}")
        return self

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)


IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")) -> list[str]:
    images = []
    if Path(dir).is_dir():
        # assert Path(dir).is_dir(), "%s is not a valid directory" % dir

        for root, _, fnames in sorted(os.walk(dir)):
            # for root, _, fnames in sorted(Path.glob(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images[: min(max_dataset_size, len(images))]
    elif Path(dir).is_file():
        images.append(dir)
        return images
    else:
        assert Path(dir).is_dir(), "%s is not a valid directory" % dir
        assert Path(dir).is_file(), "%s is not a valid file" % dir


def default_loader(path):
    return Image.open(path).convert("RGB")


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, "r") as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)
    return imlist


class ImageFilelist(data.Dataset):
    def __init__(
        self,
        root,
        flist,
        transform=None,
        flist_reader=default_flist_reader,
        loader=default_loader,
    ):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(Path(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(
        self,
        root,
        flist,
        transform=None,
        flist_reader=default_flist_reader,
        loader=default_loader,
    ):
        self.root = root
        self.imlist = flist_reader(Path(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split("/")[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [
            (impath, self.class_to_idx[impath.split("/")[0]]) for impath in self.imlist
        ]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(Path(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""


class ImageFolder(data.Dataset):
    """
    根据文件夹制作数据集
    """

    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
