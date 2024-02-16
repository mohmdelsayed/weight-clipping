from core.learner.sl.learner import Learner
from core.optim.upgd import AdaptiveUPGD, UPGD

class UPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = UPGD
        name = "upgd"
        super().__init__(name, network, optimizer, optim_kwargs)

class AdaUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaptiveUPGD
        name = "adaupgd"
        super().__init__(name, network, optimizer, optim_kwargs)