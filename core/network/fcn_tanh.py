import torch.nn as nn

class FCNTanh(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=300):
        super(FCNTanh, self).__init__()
        self.name = "fcn_tanh"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.Tanh())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units // 2))
        self.add_module("act_2", nn.Tanh())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units // 2, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name


class FCNTanhSmallWithNoBias(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=32):
        super(FCNTanhSmallWithNoBias, self).__init__()
        self.name = "fcn_tanh_small_no_bias"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units, bias=False))
        self.add_module("act_1", nn.Tanh())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units, bias=False))
        self.add_module("act_2", nn.Tanh())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units, out_features=n_outputs, bias=False))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
    
class FCNTanhSmall(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=64):
        super(FCNTanhSmall, self).__init__()
        self.name = "fcn_tanh_small"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.Tanh())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        self.add_module("act_2", nn.Tanh())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        self.add_module("act_3", nn.Tanh())
        self.add_module("linear_4", nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
    
class FCNTanhSmallSoftmax(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=64):
        super(FCNTanhSmallSoftmax, self).__init__()
        self.name = "fcn_tanh_small_softmax"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.Tanh())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        self.add_module("act_2", nn.Tanh())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        self.add_module("act_3", nn.Tanh())
        self.add_module("linear_4", nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
        self.add_module("log_softmax", nn.LogSoftmax(dim=1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name

if __name__ == "__main__":
    net = FCNTanh()
    print(net)
