from torch import nn
from typing_extensions import Literal

from .networks import define_D, define_G


class CycleGANModel(nn.Module):
    def __init__(
        self,
        input_nc=3,
        output_nc=3,
        ngf=64,
        ndf=64,
        n_layers_D=3,
        init_gain=0.02,
        use_dropout=False,
        netG: Literal[
            "resnet_9blocks", "resnet_6blocks", "unet_256", "unet_128"
        ] = "resnet_9blocks",
        netD: Literal["basic", "n_layers", "pixel"] = "basic",
        norm: Literal["instance", "batch", "AdaIN", " none"] = "instance",
        init_type: Literal["normal", "xavier", "kaiming", "orthogonal"] = "normal",
        mode: Literal["train", "test"] = "train",
    ):
        super().__init__()
        self.net_G_A = define_G(
            input_nc, output_nc, ngf, netG, norm, use_dropout, init_type, init_gain
        )

        if mode == "train":
            self.net_G_B = define_G(
                input_nc, output_nc, ngf, netG, norm, use_dropout, init_type, init_gain
            )
            self.net_D_A = define_D(
                output_nc, ndf, netD, n_layers_D, norm, init_type, init_gain
            )
            self.net_D_B = define_D(
                input_nc, ndf, netD, n_layers_D, norm, init_type, init_gain
            )

    def forward(self, a):
        return self.net_G_A(a)  # G_A(A)
