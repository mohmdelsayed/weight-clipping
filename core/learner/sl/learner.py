import torch
class Learner:
    def __init__(self, name, network, optimizer, optim_kwargs):
        self.network_cls = network
        # self.device = torch.device('mps' if torch.backends.mps.is_available else 'cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optim_kwargs = optim_kwargs
        for k, v in optim_kwargs.items():
            if isinstance(v, str):
                optim_kwargs[k] = float(v)
        self.optimizer_cls = optimizer
        self.name = name

    def __str__(self) -> str:
        return self.name

    def predict(self, input):
        output = self.network(input)
        return output

    def setup_task(self, task):
        self.network = self.network_cls(n_obs=task.n_inputs, n_outputs=task.n_outputs).to(self.device)
        self.parameters = list(self.network.parameters())
        self.named_parameters = list(self.network.named_parameters())
        self.setup_optimizer()

    def setup_optimizer(self):
        self.optimizer = self.optimizer_cls(self.parameters, **self.optim_kwargs)

    def update_params(self, closure):
        return self.optimizer.step(closure)

    def zero_grad(self):
        self.optimizer.zero_grad()