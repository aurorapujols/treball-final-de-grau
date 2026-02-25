import numpy as np
import torch
import torch.nn.functional as F

class StreamingMetrics:
    def __init__(self, alpha, t):
        self.alpha = alpha
        self.t = t
        self.align_sum = 0.0
        self.align_count = 0

        self.uniform_feats = []
        
    @torch.no_grad()
    def update(self, z_i, z_j):
        """
        Update metrics for a batch.
        z_i, z_j: embeddings of shape (B,d)
        """
        batch_align = (z_i - z_j).norm(dim=1).pow(self.alpha)
        self.align_sum += batch_align.sum().item()
        self.align_count += z_i.size(0)

        self.uniform_feats.append(z_i.detach().cpu())

    def compute(self):
        # Alignment 
        alignment = self.align_sum / self.align_count 
        
        # Uniformity 
        z = torch.cat(self.uniform_feats, dim=0) # (N, d) 
        z = F.normalize(z, dim=1) 
        sq_pdist = torch.pdist(z, p=2).pow(2) 
        uniformity = (sq_pdist.mul(-self.t).exp().mean().log().item()) 
        return alignment, uniformity


def alignment(z_i, z_j, alpha=2):
    return (z_i - z_j).norm(dim=1).pow(alpha).mean()

def uniformity(x, t=2): 
    sq_pdist=torch.pdist(x,p=2).pow(2) 
    return sq_pdist.mul(-t).exp().mean().log()

def semantic_tolerance(features, labels):
    feats = torch.tensor(features)
    labels = np.array(labels)

    tolerances = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        z = feats[idx]
        pdist = torch.pdist(z, p=2)
        tolerances.append(pdist.mean().item())

    return np.mean(tolerances)


def semantic_alignment(features, labels):
    feats = torch.tensor(features)
    labels = np.array(labels)

    alignments = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        z = feats[idx]
        pdist = torch.pdist(z, p=2).pow(2)
        alignments.append(pdist.mean().item())

    return np.mean(alignments)

def global_uniformity(features, t=2):
    x = torch.tensor(features)  # shape (N, dim)
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return (sq_pdist.mul(-t).exp().mean().log().item())


# 3. Feature norm statistics
# - mean L2 norm
# - variance of norms
# - histogram of norms
# These help detect collapse.

def compute_norm_stats(features):
    norms = np.linalg.norm(features, axis=1)
    mean_norm = norms.mean()
    var_norm = norms.var()
    hist, bin_edges = np.histogram(norms, bins=30)
    return mean_norm, var_norm, hist, bin_edges
