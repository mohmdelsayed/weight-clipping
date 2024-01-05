from core.learner.sl.learner import Learner
from core.optim.l2_init import L2Init

class L2InitLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = L2Init
        name = "l2_init"
        super().__init__(name, network, optimizer, optim_kwargs)