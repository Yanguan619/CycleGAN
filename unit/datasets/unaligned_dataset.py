import os
import random


from unit.datasets import make_dataset
from unit.datasets.base_dataset import get_transform
from PIL import Image
from typing_extensions import Literal


class UnalignedDataset:
    def __init__(
        self,
        data_root,
        input_nc=3,
        output_nc=3,
        load_size=286,
        crop_size=256,
        opt=None,
        direction="AtoB",
        max_dataset_size=float("inf"),
        phase: Literal["train", "val", "test"] = "train",
    ):
        self.opt = opt
        # create path '/data_name/trainA' '/data_name/trainB'
        self.dir_A = os.path.join(data_root, phase + "A")
        self.dir_B = os.path.join(data_root, phase + "B")
        # load images from '/data_name/trainA' '/data_name/trainB'
        self.A_paths = sorted(make_dataset(self.dir_A, max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, max_dataset_size))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        BtoA = direction == "BtoA"
        grayscale = input_nc == 1
        grayscale_out = output_nc == 1
        input_nc = output_nc if BtoA else input_nc
        output_nc = input_nc if BtoA else output_nc
        self.transform_A = get_transform(self.opt, load_size, crop_size, grayscale)
        self.transform_B = get_transform(self.opt, load_size, crop_size, grayscale_out)

    def __getitem__(self, index) -> dict:
        # make sure index is within then range
        A_path = self.A_paths[index % self.A_size]
        if hasattr(self.opt, "serial_batches"):
            # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
