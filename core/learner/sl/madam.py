from core.learner.sl.learner import Learner
from core.optim.madam import Madam

class MadamLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = Madam
        name = "madam"
        super().__init__(name, network, optimizer, optim_kwargs)
