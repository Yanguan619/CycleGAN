import os
import sys

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
import pandas as pd
import seaborn as sn

# from IPython.core.display import display
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from dataclasses import dataclass, field
from unit.datasets.unaligned_dataset import UnalignedDataset


@dataclass
class DataModule(LightningDataModule):
    data_name: str
    dims: tuple = (3, 256, 256)
    crop_size: int = 256
    load_size: int = 286
    data_dir: str = os.environ.get("PATH_DATASETS", "data/")
    batch_size: int = 1 if torch.cuda.is_available() else 1
    num_workers: int = 0  # int(os.cpu_count() / 2)
    transforms_opt: list = field(default_factory=lambda: ["resize", "crop", "flip"])

    def __post_init__(self):
        super().__init__()
        self.data_dir = os.path.join(self.data_dir, self.data_name)
        print(self.data_dir)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = UnalignedDataset(
                self.data_dir,
                opt=self.transforms_opt,
                load_size=self.load_size,
                crop_size=self.crop_size,
                phase="train",
            )
        if stage == "test" or stage is None:
            self.dataset_test = UnalignedDataset(
                self.data_dir, opt=self.transforms_opt, phase="test"
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            # num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            # num_workers=self.num_workers,
        )


# class TrainerModel(LightningModule):
#     def __init__(
#         self,
#         data_dir: str = PATH_DATASETS,
#         batch_size: int = BATCH_SIZE,
#         num_workers: int = NUM_WORKERS,
#         hidden_size=64,
#         lr=2e-4,
#     ):
#         super().__init__()
#         self.data_dir = data_dir
#         self.hidden_size = hidden_size
#         self.lr = lr
#         self.num_workers = num_workers
#         self.batch_size = batch_size

#         self.num_classes = 10
#         self.dims = (1, 28, 28)
#         channels, width, height = self.dims

#         self.transform = transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
#         )

#         self.model = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(channels * width * height, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_size, self.num_classes),
#         )
#         self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
#         self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

#     def forward(self, x: Tensor):
#         x = self.model(x)
#         return F.log_softmax(x, dim=1)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.nll_loss(logits, y)
#         self.log("train_loss", loss, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.nll_loss(logits, y)
#         pre = torch.argmax(logits, dim=1)
#         self.val_acc.update(pre, y)

#         self.log("val_loss", loss, prog_bar=True)
#         self.log("val_acc", self.val_acc, prog_bar=True)

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.nll_loss(logits, y)
#         pre = torch.argmax(logits, dim=1)
#         self.test_acc.update(pre, y)

#         self.log("test_loss", loss, prog_bar=True)
#         self.log("test_acc", self.test_acc, prog_bar=True)

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         return optimizer

#     def prepare_data(self) -> None:
#         MNIST(self.data_dir, train=True, download=False)
#         MNIST(self.data_dir, train=False, download=False)

#     def setup(self, stage: str = None):
#         if stage == "fit" or stage is None:
#             mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
#             self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
#         if stage == "test" or stage is None:
#             self.mnist_test = MNIST(
#                 self.data_dir, train=False, transform=self.transform
#             )

#     def train_dataloader(self):
#         return DataLoader(
#             self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers
#         )
