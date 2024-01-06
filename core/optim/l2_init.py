import torch

class L2Init(torch.optim.Optimizer):
    def __init__(self, params, optimizer=torch.optim.Adam, **kwargs):
        defaults = dict()
        super(L2Init, self).__init__(params, defaults)
        self.optimizer = optimizer(self.param_groups, **kwargs)
        self.param_groups = self.optimizer.param_groups
        self.defaults.update(self.optimizer.defaults)

    def step(self, closure=None):
        self.zero_grad()
        loss = closure()
        loss.backward()
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["init_weights"] = p.data.clone()
                p.data.add_(group["weight_decay"] * state["init_weights"], alpha=group["lr"])
        self.optimizer.step()
        return loss