import os

import numpy as np

from pathlib import Path

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor
from torch.autograd import Variable

from unit.models.munit import MUNITModel
from unit.utils import get_config, tensor2np
from unit.models.networks import load_vgg16, vgg_preprocess
from .base import DataModule
from . import recon_criterion

PATH_DATASETS = os.environ.get("PATH_DATASETS", "../datasets")
BATCH_SIZE = os.environ.get("BATCH_SIZE", 1)
NUM_WORKERS = os.environ.get("NUM_WORKERS", 1)


class TrainerModel(LightningModule):
    def __init__(
        self,
        model=MUNITModel(),
        cwh: tuple = (3, 128, 128),
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        lambda_identity=0.5,
        lambda_A=10,
        lambda_B=10,
        data_dir=PATH_DATASETS,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        **kwargs,
    ):
        super().__init__()
        self.hp = get_config(
            Path.cwd().joinpath("models/configs/summer2winter_yosemite.yaml")
        )
        # self.save_hyperparameters()
        self.automatic_optimization = False

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.net_G_A = model.net_G_A
        self.net_G_B = model.net_G_B
        self.net_D_A = model.net_D_A
        self.net_D_B = model.net_D_B
        # Load VGG model if needed
        # 加载VGG模型，用来计算感知 loss
        if "vgg_w" in self.hp.keys() and self.hp["vgg_w"] > 0:
            self.vgg = load_vgg16("/models")
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        else:
            self.hp["vgg_w"] = 0
        self.example_input_array = [torch.Tensor(1, *cwh)]

    def forward(self, x: Tensor):
        return self.net_G_A(x)  # G_A(A)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g_a = torch.optim.Adam(self.net_G_A.parameters(), lr=lr, betas=(b1, b2))
        opt_g_b = torch.optim.Adam(self.net_G_B.parameters(), lr=lr, betas=(b1, b2))
        opt_d_a = torch.optim.Adam(self.net_D_A.parameters(), lr=lr, betas=(b1, b2))
        opt_d_b = torch.optim.Adam(self.net_D_B.parameters(), lr=lr, betas=(b1, b2))

        return [opt_g_a, opt_g_b, opt_d_a, opt_d_b], []

    def test_step(self, batch, batch_idx):
        real_A, real_B = batch["A"], batch["B"]

        # 给输入的 real_a, real_b 加入随机噪声
        style_dim = self.hp["gen"]["style_dim"]
        style_a = Variable(torch.randn(real_A.size(0), style_dim, 1, 1))
        style_b = Variable(torch.randn(real_B.size(0), style_dim, 1, 1))
        # 编码
        content_a, _ = self.net_G_A.encode(real_A)
        content_b, _ = self.net_G_B.encode(real_B)
        # 交叉解码
        fake_A = self.net_G_A.decode(content_b, style_a)
        fake_B = self.net_G_B.decode(content_a, style_b)
        # nhwc
        img_batch = (real_A, fake_B, real_B, fake_A)
        img_batch = np.stack(tuple(map(lambda img: tensor2np(img), img_batch)), axis=0)
        img_batch = np.concatenate((img_batch[:2], img_batch[2:]), axis=1)
        self.logger.experiment.add_images(
            "img_batch", img_batch, self.global_step, dataformats="NHWC"
        )

    def training_step(self, batch, batch_idx):
        real_A, real_B = batch["A"], batch["B"]
        optimizer_g_a, optimizer_g_b, optimizer_d_a, optimizer_d_b = self.optimizers()

        # cal loss
        loss_G = self.__cal_loss_G(real_A, real_B)

        optimizer_g_a.zero_grad()
        optimizer_g_b.zero_grad()
        self.manual_backward(loss_G)
        optimizer_g_a.step()
        optimizer_g_b.step()

        self.toggle_optimizer(optimizer_d_a)
        self.toggle_optimizer(optimizer_d_b)
        loss_D = self.__cal_loss_D(real_A, real_B)

        optimizer_d_a.zero_grad()
        optimizer_d_b.zero_grad()
        self.manual_backward(loss_D)
        optimizer_d_a.step()
        optimizer_d_b.step()
        self.untoggle_optimizer(optimizer_d_a)
        self.untoggle_optimizer(optimizer_d_b)

        self.log_dict(
            {"loss_G": loss_G, "loss_D_A": loss_D},
            prog_bar=True,
            batch_size=self.batch_size,
        )

    def __cal_loss_G(self, real_A, real_B):
        """计算生成器损失"""
        # 给输入的 real_a, real_b 加入随机噪声
        style_dim = self.hp["gen"]["style_dim"]
        style_a = Variable(torch.randn(real_A.size(0), style_dim, 1, 1))
        style_b = Variable(torch.randn(real_B.size(0), style_dim, 1, 1))
        # 编码
        content_a, s_a_prime = self.net_G_A.encode(real_A)
        content_b, s_b_prime = self.net_G_B.encode(real_B)
        # 解码
        x_a_recon = self.net_G_A.decode(content_a, s_a_prime)
        x_b_recon = self.net_G_B.decode(content_b, s_b_prime)
        # 交叉解码
        x_ba = self.net_G_A.decode(content_b, style_a)
        x_ab = self.net_G_B.decode(content_a, style_b)
        # 再编码
        content_b_recon, s_a_recon = self.net_G_A.encode(x_ba)
        content_a_recon, s_b_recon = self.net_G_B.encode(x_ab)
        # 再解码
        x_aba = (
            self.net_G_A.decode(content_a_recon, s_a_prime)
            if self.hp["recon_x_cyc_w"] > 0
            else None
        )
        x_bab = (
            self.net_G_B.decode(content_b_recon, s_b_prime)
            if self.hp["recon_x_cyc_w"] > 0
            else None
        )
        # reconstruction loss，重构图片与真实图片的 loss
        loss_gen_recon_x_a = recon_criterion(x_a_recon, real_A)
        loss_gen_recon_x_b = recon_criterion(x_b_recon, real_B)
        # 合成图片再编码得到的 style code，与正态分布的随机生成的 style code 的 loss
        loss_gen_recon_s_a = recon_criterion(s_a_recon, style_a)
        loss_gen_recon_s_b = recon_criterion(s_b_recon, style_b)
        # 合成图片再编码得到的 content code，与真实图片编码得到的content code 的 loss
        loss_gen_recon_content_a = recon_criterion(content_a_recon, content_a)
        loss_gen_recon_content_b = recon_criterion(content_b_recon, content_b)
        # cycle loss
        loss_gen_cycrecon_x_a = (
            recon_criterion(x_aba, real_A) if self.hp["recon_x_cyc_w"] > 0 else 0
        )
        loss_gen_cycrecon_x_b = (
            recon_criterion(x_bab, real_B) if self.hp["recon_x_cyc_w"] > 0 else 0
        )
        # GAN loss, 最终生成图片与真实图片之间的loss
        loss_gen_adv_a = self.net_D_A.calc_gen_loss(x_ba)
        loss_gen_adv_b = self.net_D_B.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss, 使用VGG计算感知loss
        loss_gen_vgg_a = (
            self.compute_vgg_loss(self.vgg, x_ba, real_B) if self.hp["vgg_w"] > 0 else 0
        )
        loss_gen_vgg_b = (
            self.compute_vgg_loss(self.vgg, x_ab, real_A) if self.hp["vgg_w"] > 0 else 0
        )
        # total loss
        loss_G = (
            self.hp["gan_w"] * loss_gen_adv_a
            + self.hp["gan_w"] * loss_gen_adv_b
            + self.hp["recon_x_w"] * loss_gen_recon_x_a
            + self.hp["recon_s_w"] * loss_gen_recon_s_a
            + self.hp["recon_c_w"] * loss_gen_recon_content_a
            + self.hp["recon_x_w"] * loss_gen_recon_x_b
            + self.hp["recon_s_w"] * loss_gen_recon_s_b
            + self.hp["recon_c_w"] * loss_gen_recon_content_b
            + self.hp["recon_x_cyc_w"] * loss_gen_cycrecon_x_a
            + self.hp["recon_x_cyc_w"] * loss_gen_cycrecon_x_b
            + self.hp["vgg_w"] * loss_gen_vgg_a
            + self.hp["vgg_w"] * loss_gen_vgg_b
        )
        return loss_G

    def __cal_loss_D(self, real_A, real_B):
        """计算判别器损失"""
        style_dim = self.hp["gen"]["style_dim"]
        style_a = Variable(torch.randn(real_A.size(0), style_dim, 1, 1))
        style_b = Variable(torch.randn(real_B.size(0), style_dim, 1, 1))
        # 编码
        content_a, _ = self.net_G_A.encode(real_A)
        content_b, _ = self.net_G_B.encode(real_B)
        # 交叉解码，得到fake图片
        x_ba = self.net_G_A.decode(content_b, style_a)
        x_ab = self.net_G_B.decode(content_a, style_b)
        loss_D_A = self.net_D_A.calc_dis_loss(x_ba.detach(), real_A)
        loss_D_B = self.net_D_B.calc_dis_loss(x_ab.detach(), real_B)
        loss_D = self.hp["gan_w"] * loss_D_A + self.hp["gan_w"] * loss_D_B
        return loss_D

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean(
            (self.model.instancenorm(img_fea) - self.model.instancenorm(target_fea))
            ** 2
        )


def load(model, state_dict_path):
    state_dict = torch.load(state_dict_path)
    model.net_G_A.load_state_dict(state_dict["a"])
    model.net_G_B.load_state_dict(state_dict["b"])


def main(data_name="test"):
    import tensorboard

    project_name = "Munit"

    dataset = DataModule(
        data_name=data_name, load_size=256, crop_size=128, dims=(3, 128, 128)
    )
    model = TrainerModel()
    load(model, r"E:\projects\weights\munit\summer2winter_yosemite.pt")

    checkpoint_callback = ModelCheckpoint(save_weights_only=True)
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=2,
        callbacks=[TQDMProgressBar(), checkpoint_callback],
        logger=[
            # CSVLogger(save_dir="./logs/CSVLogger", name="CycleGAN", flush_logs_every_n_steps=1),
            TensorBoardLogger(
                save_dir=f"./logs/TensorBoardLogger/{project_name}", name=data_name
            ),
        ],
    )
    trainer.test(
        model=model,
        dataloaders=dataset,
        # ckpt_path=r"housecat2bigcat.ckpt",
    )


# if __name__ == "__main__":
# main()
