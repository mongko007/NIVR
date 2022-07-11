import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np

import time
import math

from utils import *
from SIREN import *


def position_encoder_t(t, lt):
    N_t = int(math.log(lt, 2) + 1)
    r_list = []
    for i in range(N_t):
        r_i = (math.sin(2 ** (i - 1) * math.pi * t), math.cos(2 ** (i - 1) * math.pi * t))
        r_list.append(r_i)

    # print(r_list)
    # print(len(r_list))

    r_numpy = np.array(r_list)
    r_tensor = torch.from_numpy(r_numpy)

    # print(r_tensor)
    # print(r_tensor.shape)

    return r_tensor


class mp(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = Siren(in_features=in_features, hidden_features=[256, 256, 256, 256, 256], hidden_layers=4,
                         out_features=out_features, outermost_linear=True)

    def forward(self, x):
        x = x.view(1, -1)
        output = self.mlp(x).view(-1, 1)
        output = torch.cat((output, output), 1)

        return output


def position_encoder_c(coords, phi_t, lc):
    N_c = int(math.log(lc, 2) + 1)
    # print(N_c)

    sin_list = []
    cos_list = []

    for i in range(N_c):
        sin_list.append(torch.sin(2 ** (i - 1) * math.pi * coords + phi_t[i]))
        cos_list.append(torch.cos(2 ** (i - 1) * math.pi * coords + phi_t[i]))
    sin_tensor = torch.stack(sin_list, 0)
    cos_tensor = torch.stack(cos_list, 0)

    # print(sin_list)
    # print(cos_list)

    # print(sin_tensor)  # Nc * 2
    # print(cos_tensor)

    # print(sin_tensor.shape)
    # print(cos_tensor.shape)

    c_tensor = torch.cat((sin_tensor, cos_tensor), dim=1)
    # print(c_tensor)
    # print(c_tensor.shape)
    c_tensor = c_tensor.view(1, -1)
    # print(c_tensor)
    # print(c_tensor.shape)

    return c_tensor


class mf(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = Siren(in_features=in_features, hidden_features=[256, 256, 256, 256, 256], hidden_layers=4,
                         out_features=out_features, outermost_linear=True)

    def forward(self, x):
        output = self.mlp(x)
        output = torch.tanh(output)

        return output


if __name__ == '__main__':
    time_len = 120
    side_len = 10
    frame_idx = 1
    t = get_t(time_len, frame_idx)
    print(t)
    r_tensor = position_encoder_t(t, time_len)
    print(r_tensor.shape)
    print(r_tensor)
    mp_in = int(math.log(time_len, 2) + 1) * 2
    mp_out = 2 * int(math.log(side_len, 2) + 1)
    print(mp_in)
    print(mp_out)
    m_p = mp(in_features=mp_in, out_features=mp_out)
    phi_t = m_p(r_tensor.float())
    print(phi_t.shape)
    print(phi_t)
    coords_list = get_mgrid(side_len=side_len, dim=2, centered=True, include_end=False)
    print('coords_list.shape：' + str(coords_list.shape))  # 65536 * 2
    coords = coords_list[1]
    print(coords.shape)
    c_tensor = position_encoder_c(coords, phi_t, side_len)
    mf_in = c_tensor.shape[1]
    mf_out = 3
    m_f = mf(in_features=mf_in, out_features=mf_out)
    output = m_f(c_tensor.float())
    print(output.shape)
    print(output)
    output_rgb = []
    for x in range(side_len):
        for y in range(side_len):
            pix_idx = side_len * x + y
            # print(pix_idx)
            c_tensor = position_encoder_c(coords_list[pix_idx], phi_t, side_len)
            output_pix = m_f(c_tensor.float())
            output_rgb.append(output_pix)
    output = torch.stack(output_rgb, -1)
    output = output.view(3, side_len, side_len)
    print('output: ' + str(output.shape))
    # a = torch.tensor([1, 2, 3])
    # b = torch.tensor([11, 22, 33])
    # c = torch.tensor([111, 222, 333])
    # d = torch.stack([a, b, c], -1)
    # print(d)


