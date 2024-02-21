import torch, math

class InitBounds:
    '''
    A class to calculate the initial bounds for weight clipping.
    Uniform Kaiming initialization bounds are used.
    Since bias requires knowledge of the previous layer's weights, we keep track of the previous weight tensor in this class.
    Linear: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L106
    Conv2d: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py#L144
    '''
    def __init__(self):
        self.previous_weight = None

    def get(self, p):
        if p.dim() == 1:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.previous_weight)
            return 1.0 / math.sqrt(fan_in)
        elif p.dim() == 2 or p.dim() == 4:
            self.previous_weight = p
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(p)
            return  1.0 / math.sqrt(fan_in)
        else:
            raise ValueError("Unsupported tensor dimension: {}".format(p.dim()))

class UPGDWithWC(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.001, beta_utility=0.999, sigma=0.001, clip_last_layer=True, zeta=1.0):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, clip_last_layer=clip_last_layer, zeta=zeta)
        super(UPGDWithWC, self).__init__(params, defaults)
        self.init_bounds = InitBounds()
    def step(self, closure=None):
        self.zero_grad()
        loss = closure()
        loss.backward()
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                bias_correction_utility = 1 - group["beta_utility"] ** state["step"]
                noise = torch.randn_like(p.grad) * group["sigma"]
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction_utility) / global_max_util)
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise) * (1-scaled_utility),
                    alpha=-group["lr"],
                )
        proportion = self.weight_clipping()
        return loss, proportion

    def weight_clipping(self):
        clipped_sum, total_sum = 0.0, 0.0
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if i >= len(group["params"])-2 and not group["clip_last_layer"]:
                    # do not clip last layer of weights/biases
                    continue
                bound = self.init_bounds.get(p)
                clipped_sum += (p.data.abs() > group["zeta"] * bound).float().sum()
                total_sum += p.data.numel()
                p.data.clamp_(-group["zeta"] * bound, group["zeta"] * bound)
        return (clipped_sum / total_sum).item()