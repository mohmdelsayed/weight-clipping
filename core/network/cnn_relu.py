
import torch.nn as nn
import torch

class CNNReLU(nn.Sequential):
    def __init__(self, n_obs=None, n_channels=3, n_outputs=10):
        super(CNNReLU, self).__init__()
        self.name = "cnn_relu"
        self.add_module("conv_1", nn.Conv2d(in_channels=n_channels, out_channels=6, kernel_size=5))
        self.add_module("pool_1", nn.MaxPool2d(kernel_size=2, stride=2))
        self.add_module("conv_2", nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5))
        self.add_module("pool_2", nn.MaxPool2d(kernel_size=2, stride=2))
        self.add_module("flatten", nn.Flatten())
        self.add_module("linear_1", nn.Linear(in_features=16 * 5 * 5, out_features=120))
        self.add_module("act_1", nn.ReLU())
        self.add_module("linear_2", nn.Linear(in_features=120, out_features=84))
        self.add_module("act_2", nn.ReLU())
        self.add_module("linear_3", nn.Linear(in_features=84, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
    

if __name__ == "__main__":

    net = CNNReLU()
    inputs = torch.randn(42, 3, 18, 32)
    output = net(inputs)
    print(output.shape)
