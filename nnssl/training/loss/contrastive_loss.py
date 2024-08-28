import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature, similarity_function, device=None):
        """
        Note: Not tested for DDP yet!
        """
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device if device is not None else torch.device("cpu")

        self.softmax = nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = (
            self._get_correlated_mask().type(torch.bool).to(self.device)
        )
        self.similarity_function = self._get_similarity_function(similarity_function)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, similarity_function):
        if similarity_function == "cosine":
            self._cosine_similarity = nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        elif similarity_function == "dot":
            return self._dot_simililarity
        else:
            raise (
                "Invalid choice of similarity function. Supported so far: (cosine/dot)."
            )

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (2N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (2N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (2N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (2N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def get_logits(self, zis, zjs):
        # Ensure inputs are on the correct device
        zis = zis.to(self.device)
        zjs = zjs.to(self.device)

        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(
            2 * self.batch_size, -1
        )

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = (
            torch.zeros(2 * self.batch_size).long().to(self.device)
        )  # TODO check if type_as works

        return logits, labels

    def forward(self, zis, zjs):

        logits, labels = self.get_logits(zis, zjs)

        loss = self.criterion(logits, labels)
        accuracy = (
            torch.max(logits.detach(), dim=1)[1] == 0
        ).sum().item() / logits.size(0)

        return loss / (2 * self.batch_size), accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example usage
    batch_size = 32
    hidden_dim = 128

    z_i = torch.randn(batch_size, hidden_dim).to(device)
    z_j = torch.randn(batch_size, hidden_dim).to(device)

    loss_fn = NTXentLoss(
        batch_size=batch_size,
        temperature=0.5,
        similarity_function="cosine",
        device=device,
    )

    loss, accuracy = loss_fn(z_i, z_j)
    print(loss)
