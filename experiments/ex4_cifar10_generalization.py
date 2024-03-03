from core.grid_search import GridSearch
from core.learner.sl.adam import AdamLearner
from core.learner.sl.weight_clipping import WeightClippingAdamLearner
from core.learner.sl.shrink_and_pertub import ShrinkAndPerturbAdamLearner
from core.network.resent18 import ResNet18
from core.runner import Runner
from core.task.cifar10_offline import CIFAR10
from core.run.sl_run_generalization import SLRunGeneralization as SLRun
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "exp4"
task = CIFAR10()
n_epochs = 300
n_seeds = 10

adam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               network=[ResNet18()],
               n_epochs=[n_epochs],
    )

shrink_and_perturb_adam_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.001],
                sigma=[0.1, 0.01, 0.001],
                weight_decay=[0.1, 0.01, 0.001],
                network=[ResNet18()],
                n_epochs=[n_epochs],
     )

weight_clipping_adam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               zeta=[1.0, 2.0, 3.0, 4.0, 5.0],
               network=[ResNet18()],
               n_epochs=[n_epochs],
    )

grids = [
        adam_grid,
        shrink_and_perturb_adam_grid,
        weight_clipping_adam_grid,
]

learners = [
    AdamLearner(),
    ShrinkAndPerturbAdamLearner(),
    WeightClippingAdamLearner(),
]

save_dir = "generated_cmds"
for learner, grid in zip(learners, grids):
    runner = Runner(SLRun(), learner, task, grid, exp_name)
    num_jobs = runner.write_cmd(save_dir)
    create_script_generator(save_dir, exp_name, learner.name, num_jobs, time="1:00:00", memory="1G")
create_script_runner(save_dir, exp_name)