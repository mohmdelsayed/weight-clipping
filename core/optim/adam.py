import torch

class Adam(torch.optim.Adam):
    def __init__(self, params, **kwargs):
        super(Adam, self).__init__(params, **kwargs)

    def step(self, closure=None):
        self.zero_grad()
        loss = closure()
        loss.backward()
        super(Adam, self).step(closure)
        return loss