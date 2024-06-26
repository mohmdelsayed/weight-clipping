from core.grid_search import GridSearch
from core.learner.sl.sgd import SGDLearner
from core.learner.sl.weight_clipping import WeightClippingSGDLearner
from core.learner.sl.shrink_and_pertub import ShrinkAndPerturbAdamLearner
from core.network.resent18 import ResNet18
from core.runner import Runner
from core.task.cifar10_offline_nonstationary import CIFAR10Nonstationary
from core.run.sl_run_generalization import SLRunGeneralization as SLRun
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "exp5"
task = CIFAR10Nonstationary()
n_epochs = 300
n_seeds = 10

sgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               network=[ResNet18()],
               n_epochs=[n_epochs],
    )

weight_clipping_sgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               zeta=[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0],
               network=[ResNet18()],
               n_epochs=[n_epochs],
    )

grids = [
        sgd_grid,
        weight_clipping_sgd_grid,
]

learners = [
    SGDLearner(),
    WeightClippingSGDLearner(),
]

save_dir = "generated_cmds"
for learner, grid in zip(learners, grids):
    runner = Runner(SLRun(), learner, task, grid, exp_name)
    num_jobs = runner.write_cmd(save_dir)
    create_script_generator(save_dir, exp_name, learner.name, num_jobs, time="1:00:00", memory="1G")
create_script_runner(save_dir, exp_name)