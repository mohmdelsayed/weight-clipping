from core.learner.sl.learner import Learner
from core.optim.shrink_and_perturb import ShrinkAndPerturb

class ShrinkAndPerturbLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = ShrinkAndPerturb
        name = "shrink_and_perturb"
        super().__init__(name, network, optimizer, optim_kwargs)