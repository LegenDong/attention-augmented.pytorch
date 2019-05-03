# -*- coding: utf-8 -*-
# @Time    : 2019/5/3 11:10
# @Author  : LegenDong
# @User    : legendong
# @File    : augmented_layer.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AugmentedLayer']


class AugmentedLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dk, dv, Nh, relative=True, fh=None, fw=None):
        super(AugmentedLayer, self).__init__()

        assert not relative or (fh is not None and fw is not None)
        assert Nh != 0 and dk % Nh == 0 and dv % Nh == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.relative = relative
        self.fh = fh
        self.fw = fw

        self.conv_layer = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size,
                                    padding=self.padding)
        self.qkv_layer = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=1)
        self.attn_layer = nn.Conv2d(self.dv, self.dv, kernel_size=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn(2 * self.fw - 1, self.dk // self.Nh) + (self.dk ** -0.5))
            self.key_rel_h = nn.Parameter(torch.randn(2 * self.fh - 1, self.dk // self.Nh) + (self.dk ** -0.5))

    def forward(self, x):
        b, c, h, w = x.size()

        conv_out = self.conv_layer(x)

        # (b, Nh, dkh or dvh, hw) and (b, Nh, dkh or dvh, h, w)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x)
        # (b, Nh, hw, dkh) matmul (b, Nh, dkh, hw) -> (b, Nh, hw, hw)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        if self.relative:
            # I change the tensor to be the same with that in paper
            # (b, Nh, hw, hw)
            w_rel_logits, h_rel_logits = self.relative_logits(q.permute(0, 1, 3, 4, 2).contiguous())
            logits += w_rel_logits
            logits += h_rel_logits
        # (b, Nh, hw)
        weights = F.softmax(logits, dim=-1)

        # (b, Nh, hw) matmul (b, Nh, hw, dvh) -> (b, Nh, hw, dvh) -> (b, dv, h, w)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = attn_out.transpose(2, 3).contiguous().view(b, self.Nh, self.dv // self.Nh, h, w)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_layer(attn_out)
        return torch.cat([conv_out, attn_out], dim=1)

    def rel_to_abs(self, x):
        # x -> (b, Nh * h, w, 2 * w - 1)
        b, Nhh, l, _ = x.size()

        col_pad = torch.zeros((b, Nhh, l, 1)).to(x.device)
        # (b, Nh * h, w, 2 * w) -> (b, Nh * h, w * 2 * w)
        x = torch.cat([x, col_pad], dim=3)
        flat_x = x.view(b, Nhh, l * 2 * l)

        flat_pad = torch.zeros((b, Nhh, l - 1)).to(x.device)
        # (b, Nh * h, w * 2 * w + w - 1) -> (b, Nh * h, w + 1, 2 * w - 1) -> (b, Nh * h, w, w)
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)
        final_x = flat_x_padded.view((b, Nhh, l + 1, 2 * l - 1))
        final_x = final_x[:, :, :l, l - 1:]
        return final_x

    def relative_logits_1d(self, q, rel_k, permute_mask):
        # the h here in q just stand for dim
        # q -> (b, Nh, h, w, dkh)
        # rel_k -> (2 * w - 1, dkh)
        b, Nh, h, w, _ = q.size()
        # (b, Nh, h, w, 2 * w - 1) -> (b, Nh, h, w, w)
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = rel_logits.view(-1, self.Nh * h, w, 2 * w - 1)
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = rel_logits.view(-1, Nh, h, w, w)

        # (b, Nh, h, h, w, w) -> (b, Nh, h * w, h * w)
        rel_logits = rel_logits.unsqueeze(dim=3)
        rel_logits = rel_logits.repeat(1, 1, 1, h, 1, 1)
        rel_logits = rel_logits.permute(permute_mask).contiguous()
        rel_logits = rel_logits.view(-1, Nh, h * w, h * w)
        return rel_logits

    def relative_logits(self, q):
        # q -> (b, Nh, h, w, dkh) -> (b, Nh, h * w, h * w)
        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, [0, 1, 2, 4, 3, 5])
        rel_logits_h = self.relative_logits_1d(q.transpose(2, 3), self.key_rel_h, [0, 1, 4, 2, 5, 3])
        return rel_logits_w, rel_logits_h

    def split_heads_2d(self, x):
        b, c, h, w = x.size()
        split = x.view(b, self.Nh, c // self.Nh, h, w)
        return split

    def combine_heads_2d(self, x):
        # x -> (b, Nh, dvh, h, w)
        b, _, _, h, w = x.size()
        combine = x.view(b, self.dv, h, w)
        return combine

    def compute_flat_qkv(self, x):
        b, c, h, w = x.size()

        # (b, c, h, w) -> (b, 2 * dk + dv, h, w)
        qkv = self.qkv_layer(x)
        # (b, dk or dv, h, w)
        q, k, v = torch.split(qkv, [self.dk, self.dk, self.dv], dim=1)
        # (b, Nh, dkh or dvh, h, w)
        q = self.split_heads_2d(q)
        k = self.split_heads_2d(k)
        v = self.split_heads_2d(v)

        dkh = self.dk // self.Nh
        q *= dkh ** -0.5

        # I think the code in paper has some problem here
        # (b, Nh, dkh or dvh, hw)
        flat_q = q.view(b, self.Nh, dkh, -1)
        flat_k = k.view(b, self.Nh, dkh, -1)
        flat_v = v.view(b, self.Nh, self.dv // self.Nh, -1)

        return flat_q, flat_k, flat_v, q, k, v


if __name__ == '__main__':
    img_data = torch.randn(16, 64, 28, 30)
    augmented_conv2d = AugmentedLayer(in_channels=64, out_channels=20, kernel_size=3, padding=1,
                                      dk=40, dv=4, Nh=4, relative=True, fh=28, fw=30)
    output = augmented_conv2d(img_data)
    print(output.size())
