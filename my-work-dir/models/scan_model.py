import torch.nn as nn

class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']    # Backbone for easier prediction (allow input images)
        self.backbone_dim = backbone['dim']
        self.nheads = nheads    # number of heads to avoid bad cluster initializations

        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])    # To allow multiple heads

    def forward(self, x, forward_pass='default'):

        # Allow to choose which forward to perform (all, only backbone, only head)
        if forward_pass == 'default':
            H = self.backbone(x)
            out = [cluster_head(H) for cluster_head in self.cluster_head]    # Create an output for each head (one tensor for each)

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            H = self.backbone(x)
            out = {'features': H, 'output': [cluster_head(H) for cluster_head in self.cluster_head]}
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out