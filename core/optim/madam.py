import torch
from torch.optim.optimizer import Optimizer

class Madam(Optimizer):

    def __init__(self, params, lr=0.01, beta=0.999, p_scale=3.0, g_bound=10.0):
        defaults = dict(lr=lr, beta=beta, g_bound=g_bound, p_scale=p_scale)
        super(Madam, self).__init__(params, defaults)

    def step(self, closure=None):
        self.zero_grad()
        loss = closure()
        loss.backward()
        clipped_sum, total_sum = 0.0, 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['max'] = group['p_scale']*(p*p).mean().sqrt().item()
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                bias_correction = 1 - group['beta'] ** state['step']
                state['exp_avg_sq'] = group['beta'] * state['exp_avg_sq'] + (1-group['beta']) * p.grad.data**2
                
                g_normed = p.grad.data / (state['exp_avg_sq']/bias_correction).sqrt()
                g_normed[torch.isnan(g_normed)] = 0
                g_normed.clamp_(-group['g_bound'], group['g_bound'])
                
                p.data *= torch.exp(-group['lr']*g_normed*torch.sign(p.data))

                clipped_sum += (p.data.abs() > state['max']).float().sum()
                total_sum += p.data.numel()
                p.data.clamp_(-state['max'], state['max'])
        return loss, clipped_sum / total_sum