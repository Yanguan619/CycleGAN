from pathlib import Path

import torch
from torch import nn
from torch.autograd import Variable
from typing_extensions import Literal

from unit.utils import get_config

from .networks import AdaINGen, MsImageDis


class MUNITModel(nn.Module):
    def __init__(
        self,
        mode: Literal["train", "test"] = "train",
    ):
        super().__init__()
        hp = get_config(
            Path.cwd().joinpath("unit/models/configs/summer2winter_yosemite.yaml")
        )
        self.net_G_A = AdaINGen(hp["input_dim_a"], hp["gen"])
        self.net_G_B = AdaINGen(hp["input_dim_b"], hp["gen"])
        if mode == "train":
            self.net_D_A = MsImageDis(hp["input_dim_a"], hp["dis"])
            self.net_D_B = MsImageDis(hp["input_dim_b"], hp["dis"])
        # 使用正则化
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hp["gen"]["style_dim"]
        # fix the noise used in sampling
        display_size = int(hp["display_size"])
        self.style_A = torch.randn(display_size, self.style_dim, 1, 1)
        self.style_B = torch.randn(display_size, self.style_dim, 1, 1)

    def forward(self, real_a):
        # 得到随机噪声
        # style_A = Variable(self.style_A)[0].unsqueeze(0)
        style_B = Variable(self.style_B)[0].unsqueeze(0)

        # 对输入图片A、B进行编码
        # 分别得到 content code 以及 style code
        content_A, style_a_fake = self.net_G_A.encode(real_a.unsqueeze(0))
        # content_B, style_b_fake = self.net_gen_b.encode(self.real_b)

        # 对 content code 加入噪声，然后进行解码（混合），得到合成图片
        fake_ab = self.net_G_B.decode(content_A, style_B)
        # self.fake_ba = self.net_gen_a.decode(content_B, style_A)
        return fake_ab
