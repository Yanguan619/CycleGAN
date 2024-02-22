import os
from dataclasses import dataclass, field
from time import sleep
import lightning

import numpy as np
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, nn

from unit.models import CycleGANModel
from unit.utils import fix_seed, tensor2np
from unit.utils.image_pool import ImagePool
from unit.utils.scorce import calc_psnr, calc_ssim

from . import GANLoss
from .base import DataModule


@dataclass(unsafe_hash=True)
class TrainerModel(LightningModule):
    cwh: list = field(default_factory=lambda: (3, 256, 256))
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999
    lambda_A: float = 10.0
    lambda_B: float = 10.0
    lambda_identity: float = 0.5
    gan_mode: str = "lsgan"
    batch_size: int = os.environ.get("BATCH_SIZE", 1)
    num_workers: int = os.environ.get("NUM_WORKERS", 0)
    plot_train_interval: int = 20

    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        model = CycleGANModel(
            # n_layers_D=8,
            # use_dropout=True,
        )
        self.net_G_A = model.net_G_A
        self.net_G_B = model.net_G_B
        self.net_D_A = model.net_D_A
        self.net_D_B = model.net_D_B
        self.example_input_array = torch.Tensor(1, *self.cwh)

        # load(self.net_G_A)
        # load(self.net_G_B)

        # loss function
        self.criterionGAN = GANLoss(self.gan_mode)  # define GAN loss.
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()
        # create image buffer to store previously generated images
        pool_size = 50
        self.fake_A_pool = ImagePool(pool_size)
        self.fake_B_pool = ImagePool(pool_size)

    def forward(self, x: Tensor):
        return self.net_G_A(x)  # G_A(A)

    def configure_optimizers(self):
        opt_G_A = torch.optim.Adam(
            self.net_G_A.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        opt_G_B = torch.optim.Adam(
            self.net_G_B.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        opt_D_A = torch.optim.Adam(
            self.net_D_A.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        opt_D_B = torch.optim.Adam(
            self.net_D_B.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )

        return [opt_G_A, opt_G_B, opt_D_A, opt_D_B], []

    def on_test_start(self) -> None:
        self.score = {"ssim_value": 0, "psnr_value": 0}
        return super().on_test_start()

    def on_test_end(self) -> None:
        print("\nInitial ssim:", self.score["ssim_value"])
        print("Initial psnr:", self.score["psnr_value"])
        return super().on_test_end()

    def test_step(self, batch, batch_idx):
        pass
        """
        real_A, real_B = batch["A"], batch["B"]
        fake_B = self.net_G_A(real_A)  # G_A(A)
        rec_A = self.net_G_B(fake_B).cpu().detach()  # G_B(G_A(A))
        fake_A = self.net_G_B(real_B)  # G_B(B)
        rec_B = self.net_G_A(fake_A).cpu().detach()  # G_A(G_B(B))

        fake_B = fake_B.cpu().detach()
        fake_A = fake_A.cpu().detach()
        # cal_score
        ssim_value = calc_ssim(tensor2np(real_A), tensor2np(fake_B))
        psnr_value = calc_psnr(tensor2np(real_A), tensor2np(fake_B))
        if self.score["ssim_value"] == 0:
            self.score["ssim_value"] = ssim_value
            self.score["psnr_value"] = psnr_value
        else:
            self.score["ssim_value"] = (self.score["ssim_value"] + ssim_value) / 2
            self.score["psnr_value"] = (self.score["psnr_value"] + psnr_value) / 2

        # n hwc
        img_batch = np.stack(
            tuple(
                map(
                    lambda img: tensor2np(img),
                    (real_A, fake_B, rec_A, real_B, fake_A, rec_B),
                )
            ),
            axis=0,
        )
        img_batch1 = np.concatenate((img_batch[0], img_batch[1], img_batch[2]), axis=1)
        img_batch2 = np.concatenate((img_batch[3], img_batch[4], img_batch[5]), axis=1)
        img_batch = np.vstack((img_batch1, img_batch2))

        self.logger.experiment.add_images(
            "img_batch", img_batch, self.global_step, dataformats="HWC"
        )
        """

    def training_step(self, batch, batch_idx):
        real_A, real_B = batch["A"], batch["B"]
        optimizer_g_a, optimizer_g_b, optimizer_d_a, optimizer_d_b = self.optimizers()

        # generate imgs
        fake_B = self.net_G_A(real_A)  # G_A(A)
        rec_A = self.net_G_B(fake_B)  # G_B(G_A(A))
        fake_A = self.net_G_B(real_B)  # G_B(B)
        rec_B = self.net_G_A(fake_A)  # G_A(G_B(B))

        ################
        # Cal loss
        ################

        # lightning 这里固定优化器后损失没有梯度了，下面判别器的不会
        # self.toggle_optimizer(optimizer_g_a)
        # self.toggle_optimizer(optimizer_g_b)
        # loss_G
        self.set_requires_grad([self.net_D_A, self.net_D_B], False)
        optimizer_g_a.zero_grad()
        optimizer_g_b.zero_grad()
        self.__cal_loss_G(real_A, real_B, fake_A, fake_B, rec_A, rec_B)
        optimizer_g_a.step()
        optimizer_g_b.step()
        # loss_D_A
        self.set_requires_grad([self.net_D_A, self.net_D_B], True)
        optimizer_d_a.zero_grad()
        self.__cal_loss_D_A(real_B, fake_B)
        optimizer_d_a.step()
        # loss_D_B
        optimizer_d_b.zero_grad()
        self.__cal_loss_D_B(real_A, fake_A)
        optimizer_d_b.step()

        ################
        # Log
        ################

        value = {
            "/".join(("loss", k)): v
            for k, v in {
                "D_A": self.loss_D_A,
                "G_A": self.loss_G_A,
                "cyc_A": self.loss_cycle_A,
                "idt_A": self.loss_idt_A,
                "D_B": self.loss_D_B,
                "G_B": self.loss_G_B,
                "cyc_B": self.loss_cycle_B,
                "idt_B": self.loss_idt_B,
            }.items()
        }
        global_step = self.global_step // 4
        self.log("log_step", self.global_step, True, logger=False)
        self.log_dict(value, logger=True, batch_size=self.batch_size)

        # n hwc
        if global_step % self.plot_train_interval == 0:
            img_batch = tuple(
                map(
                    lambda img: tensor2np(img.cpu().detach()),
                    (
                        real_A,
                        fake_B,
                        rec_A,
                        self.idt_B,
                        real_B,
                        fake_A,
                        rec_B,
                        self.idt_A,
                    ),
                )
            )

            img_batch = np.stack(img_batch, axis=0)
            img_batch1 = np.concatenate(
                (img_batch[0], img_batch[1], img_batch[2], img_batch[3]), axis=1
            )
            img_batch2 = np.concatenate(
                (img_batch[4], img_batch[5], img_batch[6], img_batch[7]), axis=1
            )
            img_batch = np.vstack((img_batch1, img_batch2))

            self.logger.experiment.add_image(
                "img_batch", img_batch, global_step, dataformats="HWC"
            )

    def __cal_loss_G(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B):
        """计算生成器损失 Cycle loss"""
        # Identity loss
        if self.lambda_identity > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.net_G_A(real_B)
            self.loss_idt_A = (
                self.criterionIdt(self.idt_A, real_B)
                * self.lambda_B
                * self.lambda_identity
            )
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.net_G_B(real_A)
            self.loss_idt_B = (
                self.criterionIdt(self.idt_B, real_A)
                * self.lambda_A
                * self.lambda_identity
            )
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.net_D_A(fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.net_D_B(fake_A), True)
        # Cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(rec_A, real_A) * self.lambda_A
        # Cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(rec_B, real_B) * self.lambda_B
        # loss+
        loss_G = (
            self.loss_G_A
            + self.loss_G_B
            + self.loss_cycle_A
            + self.loss_cycle_B
            + self.loss_idt_A
            + self.loss_idt_B
        )
        loss_G.backward()

    def __cal_back_loss_D_basic(self, net_D, real, fake):
        """计算判别器损失，即 GAN loss"""
        # Real
        pred_real = net_D(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = net_D(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # loss+
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def __cal_loss_D_A(self, real_B, fake_B):
        fake_B = self.fake_B_pool.query(fake_B)
        self.loss_D_A = self.__cal_back_loss_D_basic(self.net_D_A, real_B, fake_B)

    def __cal_loss_D_B(self, real_A, fake_A):
        fake_A = self.fake_A_pool.query(fake_A)
        self.loss_D_B = self.__cal_back_loss_D_basic(self.net_D_B, real_A, fake_A)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith("InstanceNorm") and (
            key == "running_mean" or key == "running_var"
        ):
            if getattr(module, key) is None:
                state_dict.pop(".".join(keys))
        if module.__class__.__name__.startswith("InstanceNorm") and (
            key == "num_batches_tracked"
        ):
            state_dict.pop(".".join(keys))
    else:
        patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


def load(net, state_dict_path="weights/cycle_gan/horse2zebra.pth"):
    state_dict = torch.load(state_dict_path, map_location="cpu")
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    try:
        for key in list(state_dict.keys()):
            patch_instance_norm_state_dict(state_dict, net, key.split("."))
    except:
        pass
    net.load_state_dict(state_dict, strict=False)


def main(data_name="horse2zebra", tensorboard=False):
    if tensorboard:
        import subprocess

        subprocess.Popen(("tensorboard", "--logdir", "./logs"), shell=True)
        sleep(2)
        subprocess.Popen(["start", "http://localhost:6006/"], shell=True)

    # fix_seed()
    lightning.seed_everything(77)
    # dataset
    dataset = DataModule(data_name=data_name, dims=(3, 256, 256))
    # 展示数据集
    # dataset.setup()
    # from unit.utils import show_pic
    # show_pic(dataset.train_dataloader())

    # model
    model = TrainerModel()

    # trainer
    checkpoint_callback = ModelCheckpoint(save_weights_only=True)
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=100,
        callbacks=[TQDMProgressBar(), checkpoint_callback],
        logger=[
            # CSVLogger(save_dir="./logs/CSVLogger", name="CycleGAN", flush_logs_every_n_steps=1),
            TensorBoardLogger(
                save_dir="./logs/TensorBoard/CycleGAN",
                name=data_name,
            ),
        ],
        log_every_n_steps=2,
    )
    # cli = LightningCLI(
    #     model_class=model,
    #     datamodule_class=dataset,
    #     trainer_defaults=trainer,
    #     save_config_overwrite=True,
    # )

    trainer.fit(
        model=model,
        datamodule=dataset,
        # ckpt_path=f"./logs/TensorBoard/CycleGAN/{data_name}/version_1/checkpoints/epoch=0-step=4434.ckpt",
    )


if __name__ == "__main__":
    main()
