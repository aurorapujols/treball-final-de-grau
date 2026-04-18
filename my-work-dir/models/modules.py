import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet50

def get_resnet_backbone(backbone_dim): 

    model = resnet18(weights=None) # or weights="IMAGENET1K_V1" if you want transfer learning 
    
    if backbone_dim == 2048:
        model = resnet50(weights=None)
        
    # Modify first conv layer to accept 1 channel 
    model.conv1 = nn.Conv2d( 1, 64, kernel_size=7, stride=2, padding=3, bias=False ) 
    
    # Remove the classification head (fc) 
    backbone = nn.Sequential(*list(model.children())[:-1]) # output: (B, 512, 1, 1) 
    
    return backbone

class SSLBackbone(nn.Module):
    def __init__(self, out_dim, scale_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, scale_dim, 3, padding=1),
            nn.BatchNorm2d(scale_dim),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(scale_dim, scale_dim*2, 3, padding=1),
            nn.BatchNorm2d(scale_dim*2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(scale_dim*2, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.max_pool2d(self.conv3(x), 2)

        # Global average pooling → fixed (batch, out_dim)
        x = x.mean(dim=[2, 3])
        return x

class SSLBackboneResNet(nn.Module):
    def __init__(self, res_net_dim):
        super().__init__()
        self.backbone = get_resnet_backbone(backbone_dim=res_net_dim)
        self.out_dim = res_net_dim   # ResNet-X output dimension

    def forward(self, x):
        h = self.backbone(x)        # (B, 512, 1, 1)
        h = h.squeeze(-1).squeeze(-1)  # (B, 512)
        return h

class SSLProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, projection_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, projection_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.normalize(x, dim=1)

class SSLProjectionHeadSimCLR(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True)
        )

    def forward(self, h):
        z = self.net(h)
        return F.normalize(z, dim=1)


class SCANHead(nn.Module):
    def __init__(self, input_dim, num_clusters):
        super().__init__()
        self.classifier  = nn.Linear(input_dim, num_clusters)

    def forward(self, x):
        return self.classifier(x)