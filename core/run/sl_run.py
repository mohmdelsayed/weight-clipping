import torch, sys
from core.utils import tasks, networks, learners, criterions
from core.logger import Logger
import signal
import traceback
import time
from functools import partial

def signal_handler(msg, signal, frame):
    print('Exit signal: ', signal)
    cmd, learner = msg
    with open(f'timeout_{learner}.txt', 'a') as f:
        f.write(f"{cmd} \n")
    exit(0)

class SLRun:
    def __init__(self, name='sl_run', n_samples=10000, task='stationary_mnist', exp_name='exp1', learner='sgd', save_path="logs", seed=0, network='fcn_relu', **kwargs):
        self.name = name
        self.n_samples = int(n_samples)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('mps' if torch.backends.mps.is_available else 'cpu')
        self.task = tasks[task]()

        self.exp_name = exp_name
        self.learner = learners[learner](networks[network], kwargs)
        self.logger = Logger(save_path)
        self.seed = int(seed)

    def start(self):
        torch.manual_seed(self.seed)
        losses_per_task = []
        clipping_proportion_per_task = []
        if self.task.criterion == 'cross_entropy':
            accuracy_per_task = []
        self.learner.setup_task(self.task)
        criterion = criterions[self.task.criterion]()
        losses_per_step = []
        clipping_proportion_per_step = []
        if self.task.criterion == 'cross_entropy':
            accuracy_per_step = []

        for i in range(self.n_samples):
            input, target = next(self.task)
            input, target = input.to(self.device), target.to(self.device)
            output = self.learner.predict(input)
            def closure():
                loss = criterion(output, target)
                return loss
            loss, proportion = self.learner.update_params(closure=closure)
            losses_per_step.append(loss.item())
            clipping_proportion_per_step.append(proportion)
            if self.task.criterion == 'cross_entropy':
                accuracy_per_step.append((output.argmax(dim=1) == target).float().mean().item())

            if i % self.task.change_freq == 0 and i > 0:
                avg_a = sum(accuracy_per_step) / len(accuracy_per_step)
                avg_l = sum(losses_per_step) / len(losses_per_step)
                avg_clipping = sum(clipping_proportion_per_step) / len(clipping_proportion_per_step)
                print(f"Task {i/self.task.change_freq}: loss {avg_l}, accuracy {avg_a}, clipping proportion {avg_clipping}")
                losses_per_task.append(avg_l)
                if self.task.criterion == 'cross_entropy':
                    accuracy_per_task.append(avg_a)
                clipping_proportion_per_task.append(avg_clipping)

                losses_per_step = []
                if self.task.criterion == 'cross_entropy':
                    accuracy_per_step = []
                clipping_proportion_per_step = []

        logging_data = {
                'losses': losses_per_task,
                'clipping_proportion': clipping_proportion_per_task,
                'exp_name': self.exp_name,
                'task': self.task.name,
                'learner': self.learner.name,
                'network': self.learner.network.name,
                'optimizer_hps': self.learner.optim_kwargs,
                'n_samples': self.n_samples,
                'seed': self.seed,
        }

        if self.task.criterion == 'cross_entropy':
            logging_data['accuracies'] = accuracy_per_task

        self.logger.log(**logging_data)


    def __str__(self) -> str:
        return self.name

if __name__ == "__main__":
    # start the run using the command line arguments
    ll = sys.argv[1:]
    args = {k[2:]:v for k,v in zip(ll[::2], ll[1::2])}
    run = SLRun(**args)
    cmd = f"python3 {' '.join(sys.argv)}"
    signal.signal(signal.SIGUSR1, partial(signal_handler, (cmd, args['learner'])))
    current_time = time.time()
    try:
        run.start()
        with open(f"finished_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} time_elapsed: {time.time()-current_time} \n")
    except Exception as e:
        with open(f"failed_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} \n")
        with open(f"failed_{args['learner']}_msgs.txt", "a") as f:
            f.write(f"{cmd} \n")
            f.write(f"{traceback.format_exc()} \n\n")