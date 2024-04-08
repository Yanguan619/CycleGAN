from pathlib import Path

import torch
from torch import nn
from typing_extensions import Literal

from unit.utils import get_config

from .networks import FewShotGen, GPPatchMcResDis


class FUNITModel(nn.Module):
    def __init__(
        self,
        mode: Literal["train", "test"] = "train",
    ):
        super(FUNITModel, self).__init__()
        hp = get_config(Path.cwd().joinpath("unit/models/configs/funit_animals.yaml"))
        self.hp = hp
        self.visual_names = ["fake_ba", "fake_ab"]
        self.net_names = ["gen", "dis", "gen_test"]
        self.net_G = FewShotGen(hp["gen"])
        if mode == "train":
            self.net_D = GPPatchMcResDis(hp["dis"])

    def forward(self, image, class_code):
        content_A = self.net_G.enc_content(image)
        fake_B = self.net_G.decode(content_A.unsqueeze(0), class_code.unsqueeze(0))
        return fake_B

    def test(self, content_data, class_data):
        self.eval()
        self.net_gen.eval()
        self.net_gen_test.eval()
        xa = content_data[0].cuda()
        xb = class_data[0].cuda()
        c_xa_current = self.net_gen.enc_content(xa)
        s_xa_current = self.net_gen.enc_class_model(xa)
        s_xb_current = self.net_gen.enc_class_model(xb)
        xt_current = self.net_gen.decode(c_xa_current, s_xb_current)
        xr_current = self.net_gen.decode(c_xa_current, s_xa_current)
        c_xa = self.net_gen_test.enc_content(xa)
        s_xa = self.net_gen_test.enc_class_model(xa)
        s_xb = self.net_gen_test.enc_class_model(xb)
        xt = self.net_gen_test.decode(c_xa, s_xb)
        xr = self.net_gen_test.decode(c_xa, s_xa)
        self.train()
        return xa, xr_current, xt_current, xb, xr, xt

    def translate_k_shot(self, content_data, class_data, k):
        self.eval()
        xa = content_data[0].cuda()
        xb = class_data[0].cuda()
        c_xa_current = self.gen_test.enc_content(xa)
        if k == 1:
            c_xa_current = self.gen_test.enc_content(xa)
            s_xb_current = self.gen_test.enc_class_model(xb)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        else:
            s_xb_current_before = self.gen_test.enc_class_model(xb)
            s_xb_current_after = s_xb_current_before.squeeze(-1).permute(1, 2, 0)
            s_xb_current_pool = torch.nn.functional.avg_pool1d(s_xb_current_after, k)
            s_xb_current = s_xb_current_pool.permute(2, 0, 1).unsqueeze(-1)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current

    def compute_k_style(self, style_batch, k):
        self.eval()
        style_batch = style_batch
        s_xb_before = self.net_G.enc_class_model(style_batch)
        s_xb_after = s_xb_before
        s_xb_pool = torch.nn.functional.avg_pool1d(s_xb_after, k)
        s_xb = s_xb_pool
        return s_xb
