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

        z = torch.cat([z_i, z_j], dim=0)    # tensor: (2B, d)
        z = F.normalize(z, dim=1)           # ensure dot product = cosine similarity

        sim = torch.matmul(z, z.T) / self.temperature   # compute pairwise similarities, tensor: (2B, 2B)

        mask = torch.eye(2 * batch_size, device=z.device).bool()    # to mask out self-similarities
        sim.masked_fill_(mask, -9e15)

        positives = torch.cat([             # concatenate positive dot products into a tensor of size 2B
            torch.diag(sim, batch_size),    # extract upper diagonal positive pairs
            torch.diag(sim, -batch_size)    # extract lower diagonal positive pairs
        ])

        denominator = sim.exp().sum(dim=1)  # sum over all dot products (except diagonal elements)
        loss = -torch.log(positives.exp() / denominator)    # compute the loss

        return loss.mean()      # average across all samples for the batch loss
    
class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss that takes also the labels of the samples, to consider multiple positive pairs in a batch.
    """
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j, labels):
        batch_size = z_i.size(0)    # batch size B

        z = torch.cat([z_i, z_j], dim=0)    # tensor: (2B, d)
        z = F.normalize(z, dim=1)           # ensure dot product = cosine similarity
        sim = torch.matmul(z, z.T) / self.temperature
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, -9e15)

        # Build positive pairs P(i)
        labels = torch.cat([labels, labels], dim=0) # (2B,)
        labels = labels.view(-1, 1)                 # (2B, 1)
        pos_mask = (labels == labels.T)             # (2B, 2B)    pos_mask[i] is boolean row marking all its positive pairs
        pos_mask = pos_mask.logical_and(~mask)     # remove self-pairs

        den = sim.exp().sum(dim=1)  # compute denominator for each class    (B,)
        num = (sim.exp() * pos_mask).sum(dim=1)                           # (B,)

        pos_counts = pos_mask.sum(dim=1)
        
        loss = - torch.log(num / den) / pos_counts

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