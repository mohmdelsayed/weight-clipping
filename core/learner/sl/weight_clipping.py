from core.learner.sl.learner import Learner
from core.optim.weight_clipping import WeightClippingAdam, WeightClippingSGD

class WeightClippingAdamLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = WeightClippingAdam
        name = "weight_clipping_adam"
        super().__init__(name, network, optimizer, optim_kwargs)

class WeightClippingSGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = WeightClippingSGD
        name = "weight_clipping_sgd"
        super().__init__(name, network, optimizer, optim_kwargs)