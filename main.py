import torch
import numpy as np


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        print(i)
        print(n)
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        print(r)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        print(seq)
        coord_seqs.append(seq)
        print(coord_seqs)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    print(ret)
    print('\n')
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    print(ret)
    return ret


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    make_coord((10, 10), ranges=None, flatten=True)

