from torch import nn
import torch
import numpy as np

from nnssl.training.nnsslTrainer.gvsl.gvsl_specific_modules import AffineTransformer, DeformableTransformer


class GVSLArchitecture(nn.Module):
    def __init__(self, backbone: nn.Module, num_input_channels: int, global_f_layer="", local_f_layer=""):
        super().__init__()
        self.backbone = backbone
        features_per_stage = self.backbone.encoder.output_channels

        # Maybe get num_channels from global_f_layer and local_f_layer instead of this?
        local_f_num_channels = features_per_stage[0]
        global_f_num_channels = features_per_stage[-1]

        aff_num_hidden = 256
        self.aff_conv = DoubleConv(2*global_f_num_channels, aff_num_hidden)
        self.aff_gap = nn.AdaptiveAvgPool3d(1)
        self.aff_rot = nn.Linear(aff_num_hidden, 3)
        self.aff_scale = nn.Linear(aff_num_hidden, 3)
        self.aff_trans = nn.Linear(aff_num_hidden, 3)
        self.aff_shear = nn.Linear(aff_num_hidden, 6)

        deform_num_hidden = 16
        self.deform_conv = DoubleConv(2*local_f_num_channels, deform_num_hidden)
        self.deform_flow = nn.Conv3d(deform_num_hidden, 3, 3, padding=1)

        recon_num_hidden = 16
        self.recon_head = nn.Sequential(nn.Conv3d(local_f_num_channels, recon_num_hidden, 3, padding=1),
                                        nn.GroupNorm(recon_num_hidden//4, recon_num_hidden),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv3d(recon_num_hidden, num_input_channels, 1))

        self.affine_transformer = AffineTransformer()
        self.deformable_transformer = DeformableTransformer()

        self.outputs = {}

        # In order to access the output from the second to last layer and the output from the encoder bottleneck
        # forward hooks are used. Depending on the backbone architecture, global_f_layer and local_f_layer can be
        # adjusted to direct GVSL to the correct layers from where to take the outputs from
        # ToDo: implement this
        self.backbone.encoder.stages[-1].blocks[-1].nonlin2.register_forward_hook(self.get_hook("global_f", self.outputs))
        self.backbone.decoder.stages[-1].convs[-1].all_modules[-1].register_forward_hook(self.get_hook("local_f", self.outputs))

    def affine_head(self, m, f):
        x = torch.cat([m, f], dim=1)
        x = self.aff_conv(x)
        xcor = self.aff_gap(x).flatten(start_dim=1, end_dim=4)
        rot = self.aff_rot(xcor)
        scl = self.aff_scale(xcor)
        trans = self.aff_trans(xcor)
        shear = self.aff_shear(xcor)

        rot = torch.clamp(rot, -1, 1) * (np.pi / 9)
        scl = torch.clamp(scl, -1, 1) * 0.25 + 1
        shear = torch.clamp(shear, -1, 1) * (np.pi / 18)

        affine_mat = self.get_affine_mat(rot, scl, trans, shear)
        return affine_mat


    def deformable_head(self, m, f):
        x = torch.cat([m, f], dim=1)
        sp_cor = self.deform_conv(x)
        flow = self.deform_flow(sp_cor)
        return flow

    def get_hook(self, name, dic: dict):
        def hook(model, input, output):
            dic[name] = output
        return hook

    def forward(self, A, B):
        # pass imgs through network, but acquire outputs from forward hook
        _ = self.backbone(A)
        fA_g, fA_l = self.outputs["global_f"], self.outputs["local_f"]
        _ = self.backbone(B)
        fB_g, fB_l = self.outputs["global_f"], self.outputs["local_f"]

        aff_mat_BA = self.affine_head(fB_g, fA_g)
        aff_fBA_l = self.affine_transformer(fB_l, aff_mat_BA)

        flow_BA = self.deformable_head(aff_fBA_l, fA_l)

        # registration
        warped_BA = self.deformable_transformer(self.affine_transformer(B, aff_mat_BA), flow_BA)

        recon_A = self.recon_head(fA_l)

        return recon_A, warped_BA, flow_BA


    def get_affine_mat(self, rot, scale, translate, shear):
        theta_x = rot[:, 0]
        theta_y = rot[:, 1]
        theta_z = rot[:, 2]
        scale_x = scale[:, 0]
        scale_y = scale[:, 1]
        scale_z = scale[:, 2]
        trans_x = translate[:, 0]
        trans_y = translate[:, 1]
        trans_z = translate[:, 2]
        shear_xy = shear[:, 0]
        shear_xz = shear[:, 1]
        shear_yx = shear[:, 2]
        shear_yz = shear[:, 3]
        shear_zx = shear[:, 4]
        shear_zy = shear[:, 5]

        # get affine mat for each sample in batch
        affine_mat = []
        for i in range(len(rot)):
            rot_mat_x = torch.tensor([[1, 0, 0], [0, torch.cos(theta_x[i]), -torch.sin(theta_x[i])],
                                           [0, torch.sin(theta_x[i]), torch.cos(theta_x[i])]])
            rot_mat_x = rot_mat_x[None, :, :]
            rot_mat_y = torch.tensor([[torch.cos(theta_y[i]), 0, torch.sin(theta_y[i])], [0, 1, 0],
                                           [-torch.sin(theta_y[i]), 0, torch.cos(theta_y[i])]])
            rot_mat_y = rot_mat_y[None, :, :]
            rot_mat_z = torch.tensor(
                [[torch.cos(theta_z[i]), -torch.sin(theta_z[i]), 0], [torch.sin(theta_z[i]), torch.cos(theta_z[i]), 0],
                 [0, 0, 1]])
            rot_mat_z = rot_mat_z[None, :, :]
            scale_mat = torch.tensor([[scale_x[i], 0, 0], [0, scale_y[i], 0], [0, 0, scale_z[i]]])
            scale_mat = scale_mat[None, :, :]
            shear_mat = torch.tensor(
                [[1, torch.tan(shear_xy[i]), torch.tan(shear_xz[i])], [torch.tan(shear_yx[i]), 1, torch.tan(shear_yz[i])],
                 [torch.tan(shear_zx[i]), torch.tan(shear_zy[i]), 1]])
            trans = torch.tensor([trans_x[i], trans_y[i], trans_z[i]])
            trans = trans[None, :, None]
            _affine_mat = torch.matmul(shear_mat,
                               torch.matmul(scale_mat, torch.matmul(rot_mat_z, torch.matmul(rot_mat_y, rot_mat_x))))
            _affine_mat = torch.cat([_affine_mat, trans], dim=-1)
            affine_mat.append(_affine_mat)

        affine_mat = torch.cat(affine_mat, 0)
        return affine_mat


class GVSLArchitecture_recon_only(GVSLArchitecture):
    def forward(self, A, B):
        # pass imgs through network, but acquire outputs from forward hook
        _ = self.backbone(A)
        fA_g, fA_l = self.outputs["global_f"], self.outputs["local_f"]

        recon_A = self.recon_head(fA_l)

        return recon_A

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//8, out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//8, out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
