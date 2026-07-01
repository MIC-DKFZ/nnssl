import math
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn


class GVSLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L_smooth = L_smooth
        self.L_ncc = L_ncc
        self.L_mse = L_mse

    def __call__(self, imgsA, recon_A, warped_BA, flow_BA):
        ncc_loss = self.L_ncc(warped_BA, imgsA)
        mse_loss = self.L_mse(imgsA, recon_A)
        smooth_loss = self.L_smooth(flow_BA)
        # return ncc_loss + mse_loss + smooth_loss
        return ncc_loss, mse_loss, smooth_loss


def L_smooth(s):
    dy = torch.abs(s[..., 1:, :, :] - s[..., :-1, :, :])
    dx = torch.abs(s[..., :, 1:, :] - s[..., :, :-1, :])
    dz = torch.abs(s[..., :, :, 1:] - s[..., :, :, :-1])

    dy = dy * dy
    dx = dx * dx
    dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def L_ncc(I, J, win=None):
    ndims = I.dim() - 2
    if win is None:
        win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]) #.cuda()
    pad_no = math.floor(win[0] / 2)

    stride = (1, 1, 1)
    padding = (pad_no, pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return 1 - torch.mean(cc)


def L_mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = int(np.prod(win))
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross


if __name__ == "__main__":
    pass


