import torch

class SGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super(SGD, self).__init__(params, **kwargs)

    def step(self, closure=None):
        self.zero_grad()
        loss = closure()
        loss.backward()
        super(SGD, self).step(closure)
        return loss