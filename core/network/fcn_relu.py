import torch.nn as nn
from functools import partial
import collections

class FCNReLU(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=300):
        super(FCNReLU, self).__init__()
        self.name = "fcn_relu"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.ReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units // 2))
        self.add_module("act_2", nn.ReLU())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units // 2, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
    
class FullyConnectedReLUWithHooks(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=300):
        super().__init__()
        self.name = "fcn_relu_hooks"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units, bias=False))
        self.add_module("act_1", nn.ReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units // 2, bias=False))
        self.add_module("act_2", nn.ReLU())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units // 2, out_features=n_outputs, bias=False))
        self.activations = collections.defaultdict(list)
        self.pre_activations = collections.defaultdict(list)
        self.next_grad = collections.defaultdict(list)

        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                layer.register_forward_hook(partial(self.prev_activation_hook, name))
                layer.reset_parameters()
            if isinstance(layer, nn.ReLU):
                layer.register_full_backward_hook(partial(self.next_grad_hook, name))
                layer.register_forward_hook(partial(self.activation_derivative_hook, name))

    def __str__(self):
        return self.name

    def prev_activation_hook(self, name, module, inp, out):
        self.pre_activations[name] = out[0].detach()

    def activation_derivative_hook(self, name, module, inp, out):
        self.activations[name] = (out > 0.0) * 1.0

    def next_grad_hook(self, name, module, g_inp, g_out):
        self.next_grad[name] = g_inp[0]

class FCNReLUSmallWithNoBias(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=32):
        super(FCNReLUSmallWithNoBias, self).__init__()
        self.name = "fcn_relu_small_no_bias"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units, bias=False))
        self.add_module("act_1", nn.ReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units, bias=False))
        self.add_module("act_2", nn.ReLU())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units, out_features=n_outputs, bias=False))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
    

class FCNReLUSmall(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=64):
        super(FCNReLUSmall, self).__init__()
        self.name = "fcn_relu_small"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.ReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        self.add_module("act_2", nn.ReLU())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        self.add_module("act_3", nn.ReLU())
        self.add_module("linear_4", nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
    
class FCNReLUSmallSoftmax(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=64):
        super(FCNReLUSmallSoftmax, self).__init__()
        self.name = "fcn_relu_small_softmax"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.ReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        self.add_module("act_2", nn.ReLU())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        self.add_module("act_3", nn.ReLU())
        self.add_module("linear_4", nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
        self.add_module("log_softmax", nn.LogSoftmax(dim=1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
    
if __name__ == "__main__":
    net = FCNReLU()
    print(net)
