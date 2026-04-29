import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SCANLoss(nn.Module):
    """
    SCAN Loss: Consists of a consistency loss to pull neighbors together in probability space
    and an entropy loss to avoid cluster collapse by encouraging a uniform distribution.
    """
    def __init__(self, entropy_weight=2.0):
        super().__init__()
        self.entropy_weight = entropy_weight

    def forward(self, anchors, neighbors):
        """
        Args:
            anchors: Logits for anchor images [batch_size, num_clusters]
            neighbors: Logits for neighbor images [batch_size, num_clusters]
        Returns:
            total_loss, consistency_loss, entropy_loss
        """
        # Convert logits to probabilities
        p_i = F.softmax(anchors, dim=1)
        p_j = F.softmax(neighbors, dim=1)

        # 1. Consistency Loss: dot product of probabilities (similarity in output space)
        # We want to maximize the probability that they belong to the same class.
        # Equivalent to -log(sum(p_i * p_j))
        similarity = torch.sum(p_i * p_j, dim=1)
        
        # We use a small epsilon to avoid log(0)
        consistency_loss = -torch.log(similarity + 1e-7).mean()

        # 2. Entropy Loss: Compute entropy of the mean probability distribution
        # This encourages the model to use all clusters across the batch.
        avg_probs = p_i.mean(dim=0)
        entropy_loss = -torch.sum(avg_probs * torch.log(avg_probs + 1e-7))

        # Total Loss: Maximize entropy = Subtract it from the minimization objective
        total_loss = consistency_loss - self.entropy_weight * entropy_loss

        return total_loss, consistency_loss, entropy_loss