import torch

class L2InitSGD(torch.optim.Optimizer):
    def __init__(self, params, optimizer=torch.optim.SGD, **kwargs):
        defaults = dict()
        super(L2InitSGD, self).__init__(params, defaults)
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
                state = self.state[p]
                if len(state) == 0:
                    state["init_weights"] = p.data.clone()
                p.data.add_(group["weight_decay"] * state["init_weights"], alpha=group["lr"])
        return loss, 0.0
    

class L2InitAdam(torch.optim.Optimizer):
    def __init__(self, params, optimizer=torch.optim.Adam, **kwargs):
        defaults = dict()
        super(L2InitAdam, self).__init__(params, defaults)
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
                state = self.state[p]
                if len(state) == 0:
                    state["init_weights"] = p.data.clone()
                p.data.add_(group["weight_decay"] * state["init_weights"], alpha=group["lr"])
        return loss, 0.0