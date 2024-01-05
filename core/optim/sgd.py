import torch

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5):
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        self.zero_grad()
        loss, output = closure()
        loss.backward()
        for group in self.param_groups:
            for p in group["params"]:
                p.data.add_(p.grad, alpha=-group["lr"])
        
        return loss, output