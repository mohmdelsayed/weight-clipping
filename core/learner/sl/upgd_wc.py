from core.learner.sl.learner import Learner
from core.optim.upgd_wc import UPGDWithWC

class UPGDWithWCLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = UPGDWithWC
        name = "upgd_wc"
        super().__init__(name, network, optimizer, optim_kwargs)