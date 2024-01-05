from core.learner.sl.learner import Learner
from core.optim.weight_clipping import WeightClipping

class WeightClippingLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = WeightClipping
        name = "weight_clipping"
        super().__init__(name, network, optimizer, optim_kwargs)