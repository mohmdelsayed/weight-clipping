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

class WeightClippingSGD(torch.optim.Optimizer):
    def __init__(self, params, zeta=1.0, optimizer=torch.optim.SGD, clip_last_layer=True, **kwargs):
        defaults = dict(zeta=zeta, clip_last_layer=clip_last_layer)
        super(WeightClippingSGD, self).__init__(params, defaults)
        self.optimizer = optimizer(self.param_groups, **kwargs)
        self.param_groups = self.optimizer.param_groups
        self.defaults.update(self.optimizer.defaults)
        self.init_bounds = InitBounds()

    def step(self, closure=None):
        self.zero_grad()
        loss = closure()
        loss.backward()
        self.optimizer.step()
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
        return clipped_sum / total_sum

class WeightClippingAdam(torch.optim.Optimizer):
    def __init__(self, params, zeta=1.0, optimizer=torch.optim.Adam, clip_last_layer=True, **kwargs):
        defaults = dict(zeta=zeta, clip_last_layer=clip_last_layer)
        super(WeightClippingAdam, self).__init__(params, defaults)
        self.optimizer = optimizer(self.param_groups, **kwargs)
        self.param_groups = self.optimizer.param_groups
        self.defaults.update(self.optimizer.defaults)
        self.init_bounds = InitBounds()

    def step(self, closure=None):
        self.zero_grad()
        loss = closure()
        loss.backward()
        self.optimizer.step()
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
        return clipped_sum / total_sum
