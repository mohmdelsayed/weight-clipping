from core.learner.sl.learner import Learner
from core.optim.l2_init import L2InitAdam, L2InitSGD

class L2InitSGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = L2InitSGD
        name = "l2_init_sgd"
        super().__init__(name, network, optimizer, optim_kwargs)

class L2InitAdamLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = L2InitAdam
        name = "l2_init_adam"
        super().__init__(name, network, optimizer, optim_kwargs)