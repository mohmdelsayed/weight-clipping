from core.learner.sl.learner import Learner
from core.optim.adam import Adam

class AdamLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = Adam
        name = "adam"
        super().__init__(name, network, optimizer, optim_kwargs)
