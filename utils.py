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


def get_mgrid(side_len, dim=2, centered=True, include_end=False):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''

    if isinstance(side_len, int):
        side_len = dim * (side_len,)
        # print(sidelen)

    if include_end:
        denom = [s - 1 for s in side_len]
    else:
        denom = (side_len[0] / 2, side_len[1] / 2)

    # print(denom)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:side_len[0], :side_len[1]], axis=-1)[None, ...].astype(np.float32)
        # print(pixel_coords)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / denom[0]
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / denom[1]
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:side_len[0], :side_len[1], :side_len[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / denom[0]
        pixel_coords[..., 1] = pixel_coords[..., 1] / denom[1]
        pixel_coords[..., 2] = pixel_coords[..., 2] / denom[2]
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    if centered:
        pixel_coords -= 1

    # print(pixel_coords)

    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)

    # print(pixel_coords.shape)
    # print(pixel_coords)

    return pixel_coords


def get_t(time_len, idx):
    t0 = -1
    t1 = 1
    frame_idx = (t1 - t0) / time_len * idx
    frame_idx -= 1
    frame_idx = np.float32(frame_idx)

    # print(frame_idx)

    return frame_idx


if __name__ == '__main__':
    get_mgrid(sidelen=10, dim=2, centered=True, include_end=False)
    # for i in range(120):
    #     get_t(120, i)
