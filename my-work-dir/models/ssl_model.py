import torch
import torch.nn as nn

from .modules import SSLBackbone, SSLBackboneResNet, SSLProjectionHead, SSLProjectionHeadSimCLR

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
        z = self.projector(h)    # (B, projection_dim)
        return h, z
