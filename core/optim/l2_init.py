import torch

class L2Init(torch.optim.Optimizer):
    def __init__(self, params, weight_decay=0.01, lr=1e-5):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(L2Init, self).__init__(params, defaults)

    def step(self, closure=None):
        self.zero_grad()
        loss, output = closure()
        loss.backward()
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["init_wegihts"] = p.data.clone()
                p.data.mul_(1 - group["weight_decay"] * group["lr"])
                p.data.add_(p.grad - group["weight_decay"] * state["init_wegihts"], alpha=-group["lr"])
        return loss, output