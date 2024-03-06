from core.grid_search import GridSearch
from core.learner.sl.sgd import SGDLearner
from core.network.resent18 import ResNet18
from core.runner import Runner
from core.task.cifar10_offline_nonstationary import CIFAR10Nonstationary
from core.run.sl_run_generalization_clipping import SLRunGeneralizationClipping as SLRun
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "exp6"
task = CIFAR10Nonstationary()
n_epochs = 300
n_seeds = 10

sgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               network=[ResNet18()],
               n_epochs=[n_epochs],
    )

grids = [
        sgd_grid,
]

learners = [
    SGDLearner(),
]

save_dir = "generated_cmds"
for learner, grid in zip(learners, grids):
    runner = Runner(SLRun(), learner, task, grid, exp_name)
    num_jobs = runner.write_cmd(save_dir)
    create_script_generator(save_dir, exp_name, learner.name, num_jobs, time="1:00:00", memory="1G")
create_script_runner(save_dir, exp_name)