import torch
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, n_obs=3*32*32, n_outputs=10):
        super(ResNet18, self).__init__()
        self.name = "resnet18"
        self.resnet18 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', weights=None)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=n_outputs)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        preferences = self.resnet18(x)
        return torch.nn.functional.softmax(preferences, dim=1)

    def __str__(self):
        return self.name