import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from im2scene.layers import ResnetBlock

'''
ACKNOWLEDGEMENT: This code is largely adopted from:
https://github.com/LMescheder/GAN_stability
'''


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


class Generator(nn.Module):
    """
        生成器网络
    """
    def __init__(self, device, z_dim, prior_dist, size=64, nfilter=16,
                 nfilter_max=512, **kwargs):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        self.device = device
        self.z_dim = z_dim
        # 随机噪声函数
        self.prior_dist = prior_dist

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)
        # dense
        self.fc = nn.Linear(z_dim, self.nf0*s0*s0)
        # 构建Resnet
        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf),
        ]
        # resnet
        self.resnet = nn.Sequential(*blocks)
        # 卷积层
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def sample_z(self, to_device=True):
        # 生成随机噪声
        z = self.prior_dist()
        if to_device:
            z = z.to(self.device)
        return z

    def forward(self, z):
        # 如果z为空
        if z is None:
            # 生成随机噪声
            z = self.prior_dist().to(self.device)
        # 获取batchsize
        batch_size = z.size(0)
        # linear
        out = self.fc(z)
        # reshape
        out = out.view(batch_size, self.nf0, self.s0, self.s0)
        # resnet
        out = self.resnet(out)
        # 卷积
        out = self.conv_img(actvn(out))
        # 双曲正切
        out = torch.tanh(out)

        return out
