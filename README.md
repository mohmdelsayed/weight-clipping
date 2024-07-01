# Weight-Clipping

The official repo for reproducing the experiments and weight clipping implementation. You can find the paper from [this link](). Here we give a minimal implementation for weight clipping with SGD (change `torch.optim.SGD` to torch.optim.Adam` if you want to use Adam).

```python
import torch, math

class InitBounds:
    def __init__(self):
        self.previous_weight = None
    def get(self, p):
        if p.dim() == 1:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.previous_weight)
            return 1.0 / math.sqrt(fan_in)
        elif p.dim() == 2 or p.dim() == 4:
            self.previous_weight = p
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(p)
            return  1.0 / math.sqrt(fan_in)
        else:
            raise ValueError("Unsupported tensor dimension: {}".format(p.dim()))

class WeightClippingSGD(torch.optim.Optimizer):
    def __init__(self, params, kappa=1.0, optimizer=torch.optim.SGD, **kwargs):
        defaults = dict(kappa=kappa)
        super(WeightClippingSGD, self).__init__(params, defaults)
        self.optimizer = optimizer(self.param_groups, **kwargs)
        self.param_groups = self.optimizer.param_groups
        self.defaults.update(self.optimizer.defaults)
        self.init_bounds = InitBounds()
    def step(self, closure=None):
        self.zero_grad()
        loss = closure()
        loss.backward()
        self.optimizer.step()
        self.weight_clipping()
    def weight_clipping(self):
        for group in self.param_groups:
            for p in group["params"]:
                bound = self.init_bounds.get(p)
                p.data.clamp_(-group["kappa"] * bound, group["kappa"] * bound)
```

## Reproducing results:
#### 1. You need to have environemnt with python 3.11:
``` sh
conda create --name torch python==3.11
conda activate torch
```
#### 2. Install Dependencies:
```sh
python -m pip install --upgrade pip
pip install .
```
#### 3. TBD

## License
Distributed under the [MIT License](https://opensource.org/licenses/MIT). See `LICENSE` for more information.



## How to cite

#### Bibtex:
```bibtex
@inproceedings{elsayed2024weightclipping,
  title={Weight clipping for deep continual and reinforcement learning},
  author={Elsayed, Mohamed and Lan, Qingfeng and Lyle, Clare and Mahmood, A Rupam},
  booktitle={Reinforcement Learning Conference},
  year={2024}
}
```

#### APA:
Elsayed, M., Lan, Q., Lyle, C., Mahmood, A. R. (2024). Weight clipping for deep continual and reinforcement learning. <em>In the First Reinforcement Learning Conference</em>
