import torch, math

def init_bounds(p):
    if p.dim() == 1:
        return 1
    return math.sqrt(1 / (p.shape[0]+p.shape[1]))

class WeightClipping(torch.optim.Optimizer):
    def __init__(self, params, beta=1.0, init_bounds=init_bounds, optimizer=torch.optim.Adam, clip_last_layer=True, **kwargs):
        defaults = dict(beta=beta, init_bounds=init_bounds, clip_last_layer=clip_last_layer)
        super(WeightClipping, self).__init__(params, defaults)
        self.optimizer = optimizer(self.param_groups, **kwargs)
        self.param_groups = self.optimizer.param_groups
        self.defaults.update(self.optimizer.defaults)

    def step(self, closure=None):
        self.zero_grad()
        loss, output = closure()
        loss.backward()
        self.optimizer.step()
        self.weight_clipping()
        return loss, output

    def weight_clipping(self):
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if i >= len(group["params"])-2 and not group["clip_last_layer"]:
                    # do not clip last layer of weights/biases
                    continue
                p.data.clamp_(-group["beta"] * group["init_bounds"](p), group["beta"] * group["init_bounds"](p))
