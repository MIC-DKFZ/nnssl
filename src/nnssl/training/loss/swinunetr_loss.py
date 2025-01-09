import torch
from torch.nn import functional as F
from torch import nn


class ContrastLoss(nn.Module):
    def __init__(self, batch_size, device, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(device))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool).to(device)).float())

    def forward(self, x):
        z = F.normalize(x, dim=1)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)


class SwinUNETRLoss(nn.Module):
    def __init__(self, batch_size, device, rec_loss_weight, contrast_loss_weight, rot_loss_weight):
        super().__init__()
        self.rec_loss = nn.L1Loss().to(device)
        self.contrast_loss = ContrastLoss(batch_size, device).to(device)
        self.rot_loss = nn.CrossEntropyLoss().to(device)

        self.rec_loss_weight = rec_loss_weight
        self.contrast_loss_weight = contrast_loss_weight
        self.rot_loss_weight = rot_loss_weight

    def __call__(self, rotations_pred, rotations, contrast, reconstructions, imgs_rotated):
        rec_loss = self.rec_loss(reconstructions, imgs_rotated)
        contrast_loss = self.contrast_loss(contrast)
        rot_loss = self.rot_loss(rotations_pred, rotations)
        return (self.rec_loss_weight * rec_loss
                + self.contrast_loss_weight * contrast_loss
                + self.rot_loss_weight * rot_loss)


if __name__ == "__main__":
    pass


