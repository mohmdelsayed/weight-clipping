from core.learner.sl.learner import Learner
from core.optim.shrink_and_perturb import ShrinkAndPerturbAdam, ShrinkAndPerturbSGD

class ShrinkAndPerturbAdamLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = ShrinkAndPerturbAdam
        name = "shrink_and_perturb_adam"
        super().__init__(name, network, optimizer, optim_kwargs)

class ShrinkAndPerturbSGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = ShrinkAndPerturbSGD
        name = "shrink_and_perturb_sgd"
        super().__init__(name, network, optimizer, optim_kwargs)