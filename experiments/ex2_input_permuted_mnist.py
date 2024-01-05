from core.grid_search import GridSearch
from core.learner.sl.sgd import SGDLearner
from core.network.fcn_relu import FCNReLU
from core.runner import Runner
from core.task.input_permuted_mnist import InputPermutedMNIST
from core.run.sl_run import SLRun
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "exp2"
task = InputPermutedMNIST()

sgd_grid = GridSearch(
               seed=[i for i in range(0, 20)],
               lr=[10 ** -i for i in range(1, 5)],
               network=[FCNReLU()],
               n_samples=[1000000],
    )


grids = [
        sgd_grid,
        sgd_grid,
]

learners = [
    SGDLearner(),
    SGDLearner(),
]

save_dir = "generated_cmds"
for learner, grid in zip(learners, grids):
    runner = Runner(SLRun(), learner, task, grid, exp_name)
    num_jobs = runner.write_cmd(save_dir)
    create_script_generator(save_dir, exp_name, learner.name, num_jobs, time="1:00:00", memory="1G")
create_script_runner(save_dir, exp_name)