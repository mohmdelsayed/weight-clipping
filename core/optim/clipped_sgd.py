import torch, math

def init_factor(p):
    if p.dim() == 1:
        return 1
    return math.sqrt(1 / (p.shape[0]+p.shape[1]))

class ClippedSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta=3.0):
        defaults = dict(lr=lr, beta=beta)
        super(ClippedSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        self.zero_grad()
        loss, output = closure()
        loss.backward()
        for group in self.param_groups:
            for p in group["params"]:
                p.data.add_(p.grad, alpha=-group["lr"])
                p.data.clamp_(-group["beta"] * init_factor(p), group["beta"] * init_factor(p))
        return loss, output