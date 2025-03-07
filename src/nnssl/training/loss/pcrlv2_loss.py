import random

import torch
from torch.nn import functional as F
from torch import nn


class PCRLv2Loss(nn.Module):
    def __init__(self, num_mid_stages: int, num_locals: int):
        super().__init__()

        self.num_mid_stages = num_mid_stages
        self.num_locals = num_locals

        self.cosine_sim = F.cosine_similarity
        self.mse = F.mse_loss

    def get_rand_stage(self):
        return random.randint(0, self.num_mid_stages-1)

    def pcrl_cosine_sim(self, embeddings_1, embeddings_2):
        rand_idx = self.get_rand_stage()
        emb_1 = embeddings_1[rand_idx]
        emb_2 = embeddings_2[rand_idx]
        return -0.5 * (self.cosine_sim(emb_1[1], emb_2[0].detach()).mean()
                       + self.cosine_sim(emb_2[1], emb_1[0].detach()).mean())

    def __call__(self, recon_A, mid_stage_recon_A, gt_recon_A, embeddings_A, embeddings_B, local_embeddings):
        recon_loss = self.mse(recon_A, gt_recon_A)

        rand_idx = self.get_rand_stage()
        mid_recon_loss = self.mse(mid_stage_recon_A[rand_idx], gt_recon_A)
        global_sim_loss = self.pcrl_cosine_sim(embeddings_A, embeddings_B)

        local_sim_loss = 0
        for i in range(self.num_locals):
            _local_embeddings = [t[:, ::self.num_locals] for t in local_embeddings]
            local_sim_loss += (self.pcrl_cosine_sim(embeddings_A, _local_embeddings)
                               + self.pcrl_cosine_sim(embeddings_B, _local_embeddings))

        # return recon_loss + mid_recon_loss + global_sim_loss + local_sim_loss
        return recon_loss , mid_recon_loss , global_sim_loss , local_sim_loss

if __name__ == "__main__":
    pass


