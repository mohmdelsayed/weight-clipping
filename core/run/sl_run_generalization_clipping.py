import torch, sys
from core.utils import tasks, networks, learners, criterions
from core.logger import Logger
from core.optim.weight_clipping import InitBounds
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

class SLRunGeneralizationClipping:
    def __init__(self, name='sl_run_generalization_clipping', n_epochs=700, task='cifar10', exp_name='exp5', learner='sgd', save_path="logs", seed=0, network='fcn_relu', **kwargs):
        self.name = name
        self.n_epochs = int(n_epochs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('mps' if torch.backends.mps.is_available else 'cpu')
        self.task = tasks[task](n_epochs=self.n_epochs)

        self.exp_name = exp_name
        self.learner = learners[learner](networks[network], kwargs)
        self.logger = Logger(save_path)
        self.seed = int(seed)

    def start(self):
        torch.manual_seed(self.seed)
        losses_per_task, accuracy_per_task, test_accuracy_per_task, test_loss_per_task = [], [], [], []
        losses_per_step, accuracy_per_step, test_accuracy_per_step, test_losses_per_step = [], [], [], []

        self.learner.setup_task(self.task)
        criterion = criterions[self.task.criterion]()

        n_samples = self.task.n_samples
        init_bounds = InitBounds()

        for i in range(n_samples+1):
            input, target = next(self.task)
            input, target = input.to(self.device), target.to(self.device)
            output = self.learner.predict(input)
            def closure():
                loss = criterion(output, target)
                return loss
            loss, _ = self.learner.update_params(closure=closure)
            losses_per_step.append(loss.item())
            accuracy_per_step.append((output.argmax(dim=1) == target).float().mean().item())

            if i % self.task.n_iteration_per_epoch == 0 and i > 0:
                avg_a = sum(accuracy_per_step) / len(accuracy_per_step)
                avg_l = sum(losses_per_step) / len(losses_per_step)
                losses_per_task.append(avg_l)
                accuracy_per_task.append(avg_a)
                losses_per_step, accuracy_per_step = [], []
                
                self.learner.network.eval()
                with torch.no_grad():
                    test_loader = self.task.get_test_dataloader()
                    for input, target in test_loader:
                        input, target = input.to(self.device), target.to(self.device)
                        output = self.learner.predict(input)
                        test_accuracy_per_step.append((output.argmax(dim=1) == target).float().mean().item())
                        test_losses_per_step.append(criterion(output, target).item())
                    avg_a_test = sum(test_accuracy_per_step) / len(test_accuracy_per_step)
                    avg_l_test = sum(test_losses_per_step) / len(test_losses_per_step)
                    test_accuracy_per_task.append(avg_a_test)
                    test_loss_per_task.append(avg_l_test)
                    test_accuracy_per_step, test_losses_per_step = [], []
                self.learner.network.train()

                print(f"Epoch {i/self.task.n_iteration_per_epoch}: loss {avg_l}, train accuracy {avg_a}, test loss {avg_l_test}, test accuracy {avg_a_test}")

            if i == n_samples // 2:
                for p in self.learner.network.parameters():
                    bound = init_bounds.get(p)
                    p.data.clamp_(-bound, bound)

        logging_data = {
                'accuracies': accuracy_per_task,
                'test_accuracies': test_accuracy_per_task,
                'test_losses': test_loss_per_task,
                'losses': losses_per_task,
                'exp_name': self.exp_name,
                'task': self.task.name,
                'learner': self.learner.name,
                'network': self.learner.network.name,
                'optimizer_hps': self.learner.optim_kwargs,
                'n_samples': n_samples,
                'n_epochs': self.n_epochs,
                'seed': self.seed,
        }

        self.logger.log(**logging_data)

    def __str__(self) -> str:
        return self.name

if __name__ == "__main__":
    # start the run using the command line arguments
    ll = sys.argv[1:]
    args = {k[2:]:v for k,v in zip(ll[::2], ll[1::2])}
    run = SLRunGeneralizationClipping(**args)
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