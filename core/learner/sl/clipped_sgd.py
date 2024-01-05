from core.learner.sl.learner import Learner
from core.optim.clipped_sgd import ClippedSGD

class ClippedSGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = ClippedSGD
        name = "clipped_sgd"
        super().__init__(name, network, optimizer, optim_kwargs)