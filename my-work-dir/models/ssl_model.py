import torch
import numpy as np
import torch.nn as nn

from PIL import Image

from .modules import SSLBackbone, SSLBackboneResNet, SSLProjectionHead, SSLProjectionHeadSimCLR
from transformations.transform import base_transform
from transformations.augment import ControlledAugmentGPU

def get_model(filepath):
    model = SSLResNet(
        res_net_dim=512,
        projection_dim=256
    )
    state = torch.load(filepath, map_location="cpu", weights_only=True) 
    model.load_state_dict(state) 
    model.eval()
    return model

def get_encoding_and_projection(model, dataloader, device):
    model.eval()

    H = []
    Z = []
    labels = []

    with torch.no_grad():
        for imgs, fnames, _, _, lbls in dataloader:
            h, z = model.project_and_encode(imgs)

            H.append(h.cpu().numpy())
            Z.append(z.cpu().numpy())

            labels.ectend(lbls)
    
    return (
        np.vstack(H),   # X_backbone
        np.vstack(Z),   # X_projection
        np.array(labels)
    )

def get_two_augmentations_projection(model, dataloader, device):
    model.eval()

    Z_i = []        # projection of augmentation 1
    Z_j = []        # projection of augmentation 2
    labels = []

    with torch.no_grad():
        for x_i, x_j, lbls in dataloader:
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            # First augmented view
            _, z_i = model.project(x_i)

            # Second augmented view
            _, z_j = model.project(x_j)

            # Alignment uses both
            Z_i.append(z_i.cpu().numpy())
            Z_j.append(z_j.cpu().numpy())

            labels.extend(lbls)

    return (
        np.vstack(Z_i),        # X_projection_head_i
        np.vstack(Z_j),        # X_projection_head_j
        np.array(labels)
    )

def compute_augmentations_distance(Z_i, Z_j):
    return np.linalg.norm(Z_i - Z_j, axis=1)

class SSLModel(nn.Module):
    def __init__(self, backbone_dim, hidden_dim, projection_dim):
        super().__init__()
        self.backbone = SSLBackbone(out_dim=backbone_dim, hidden_dim=hidden_dim)
        self.projection_head = SSLProjectionHead(in_dim=backbone_dim, hidden_dim=backbone_dim * 2, projection_dim=projection_dim)

    def forward(self, x):
        h = self.backbone(x)              # (batch, backbone_dim)
        z = self.projection_head(h)       # (batch, proj_dim)
        return h, z
    
class SSLResNet(nn.Module):
    def __init__(self, res_net_dim=512, projection_dim=128):
        super().__init__()
        self.encoder = SSLBackboneResNet(res_net_dim=res_net_dim)
        self.projector = SSLProjectionHeadSimCLR(in_dim=res_net_dim, out_dim=projection_dim)

    def forward(self, x):
        h = self.encoder(x)      # (B, 512)
        z = self.projector(h)    # (B, 256)
        return h, z
    
    def encode(self, x):
        self.eval()
        with torch.no_grad():
            h = self.encoder(x)
        return h
    
    def project(self, h):
        self.eval()
        with torch.no_grad():
            z = self.projector(h)
        return z
    
    def encode_and_project(self, x):
        self.eval()
        with torch.no_grad():
            h = self.encoder(x)
            z = self.projector(h)
        return h, z
