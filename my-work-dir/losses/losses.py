import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.metrics import uniformity, alignment

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss that brings embedding of positive paris together. Uses dynamic batch size to
    handle different size of last batch.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)

        sim = torch.matmul(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, -9e15)

        positives = torch.cat([
            torch.diag(sim, batch_size),
            torch.diag(sim, -batch_size)
        ])

        denominator = sim.exp().sum(dim=1)
        loss = -torch.log(positives.exp() / denominator)

        return loss.mean()

def scan_loss(logits, nn_logits, temperature=0.1):
    # Softmax
    p = torch.softmax(logits / temperature, dim=1)
    q = torch.softmax(nn_logits / temperature, dim=1)

    # Consistency loss (KL divergence)
    consistency = torch.mean(torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10)), dim=1))

    # Entropy minimization (encourage confident predictions)
    entropy_min = -torch.mean(torch.sum(p * torch.log(p + 1e-10), dim=1))

    # Entropy maximization (encourage cluster diversity)
    avg_p = torch.mean(p, dim=0)
    entropy_max = torch.sum(avg_p * torch.log(avg_p + 1e-10))

    return consistency + 0.1 * entropy_min + 0.1 * entropy_max