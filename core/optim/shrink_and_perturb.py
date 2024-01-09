import torch

class ShrinkAndPerturb(torch.optim.Optimizer):
    def __init__(self, params, sigma=0.01, optimizer=torch.optim.Adam, **kwargs):
        defaults = dict(sigma=sigma)
        super(ShrinkAndPerturb, self).__init__(params, defaults)
        self.optimizer = optimizer(self.param_groups, **kwargs)
        self.param_groups = self.optimizer.param_groups
        self.defaults.update(self.optimizer.defaults)

    def step(self, closure=None):
        self.zero_grad()
        loss = closure()
        loss.backward()
        self.optimizer.step()
        for group in self.param_groups:
            for p in group["params"]:
                perturbation = torch.randn_like(p.data) * group["sigma"]
                p.data.add_(perturbation, alpha=-group["lr"])
        return loss