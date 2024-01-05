from core.learner.sl.learner import Learner
from core.optim.sgd import SGD


class SGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SGD
        name = "sgd"
        super().__init__(name, network, optimizer, optim_kwargs)