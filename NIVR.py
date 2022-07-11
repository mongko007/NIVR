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
from modules import *


class NIVR(nn.Module):
    def __init__(self, side_len, time_len, mp_in, mp_out, mf_in, mf_out):
        super().__init__()
        self.mp = mp(in_features=mp_in, out_features=mp_out)
        self.mf = mf(in_features=mf_in, out_features=mf_out)
        self.side_len = side_len
        self.time_len = time_len

    def forward(self, x):
        frame_idx = x['idx']
        coords = x['coords']
        t = get_t(self.time_len, frame_idx)
        r_tensor = position_encoder_t(t, self.time_len)
        phi_t = self.mp(r_tensor.float())
        output_rgb = []
        for x in range(self.side_len):
            for y in range(self.side_len):
                pix_idx = self.side_len * x + y
                c_tensor = position_encoder_c(coords[pix_idx], phi_t, self.side_len)
                output_pix_rgb = self.mf(c_tensor.float())
                output_rgb.append(output_pix_rgb)
        output = torch.stack(output_rgb, -1)
        output = output.view(3, self.side_len, self.side_len)

        return output
