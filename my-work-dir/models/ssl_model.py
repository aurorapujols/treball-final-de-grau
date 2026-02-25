import torch
import torch.nn as nn

from .modules import SSLBackbone, SSLBackboneResNet, SSLProjectionHead

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
    def __init__(self, backbone_dim, hidden_dim, projection_dim):
        super().__init__()
        self.encoder = SSLBackboneResNet(out_dim=backbone_dim)
        self.projector = SSLProjectionHead(in_dim=backbone_dim, hidden_dim=hidden_dim, projection_dim=projection_dim)

    def forward(self, x):
        h = self.encoder(x)      # (B, 512)
        z = self.projector(h)    # (B, projection_dim)
        return h, z
