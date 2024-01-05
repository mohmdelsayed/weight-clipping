import torch.nn as nn

class FCNLeakyReLU(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=300):
        super(FCNLeakyReLU, self).__init__()
        self.name = "fcn_leakyrelu"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.LeakyReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units // 2))
        self.add_module("act_2", nn.LeakyReLU())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units // 2, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name


class FCNLeakyReLUSmallWithNoBias(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=32):
        super(FCNLeakyReLUSmallWithNoBias, self).__init__()
        self.name = "fcn_leakyrelu_small_no_bias"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units, bias=False))
        self.add_module("act_1", nn.LeakyReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units, bias=False))
        self.add_module("act_2", nn.LeakyReLU())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units, out_features=n_outputs, bias=False))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
    

class FCNLeakyReLUSmall(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=64):
        super(FCNLeakyReLUSmall, self).__init__()
        self.name = "fcn_leakyrelu_small"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.LeakyReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        self.add_module("act_2", nn.LeakyReLU())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        self.add_module("act_3", nn.LeakyReLU())
        self.add_module("linear_4", nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
    
class FCNLeakyReLUSmallSoftmax(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=64):
        super(FCNLeakyReLUSmallSoftmax, self).__init__()
        self.name = "fcn_leakyrelu_small_softmax"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.LeakyReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        self.add_module("act_2", nn.LeakyReLU())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        self.add_module("act_3", nn.LeakyReLU())
        self.add_module("linear_4", nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
        self.add_module("log_softmax", nn.LogSoftmax(dim=1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
    
if __name__ == "__main__":
    net = FCNLeakyReLU()
    print(net)
