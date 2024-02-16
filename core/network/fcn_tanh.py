import torch
import torch.nn as nn
from functools import partial
import collections

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

class FCNTanhWithHooks(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=300):
        super().__init__()
        self.name = "fcn_tanh_with_hooks"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.Tanh())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units // 2))
        self.add_module("act_2", nn.Tanh())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units // 2, out_features=n_outputs))
        self.activations = collections.defaultdict(list)
        self.n_units = n_hidden_units + n_hidden_units // 2

        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
            if isinstance(layer, nn.Tanh):
                layer.register_forward_hook(partial(self.activation_hook, name))

    def __str__(self):
        return self.name

    def activation_hook(self, name, module, inp, out):
        self.activations[name] = torch.sum(out.abs() > 0.9).item()