from core.grid_search import GridSearch
from core.learner.sl.sgd import SGDLearner
from core.learner.sl.adam import AdamLearner
from core.learner.sl.l2_init import L2InitSGDLearner, L2InitAdamLearner
from core.learner.sl.madam import MadamLearner
from core.learner.sl.shrink_and_pertub import ShrinkAndPerturbSGDLearner, ShrinkAndPerturbAdamLearner
from core.learner.sl.weight_clipping import WeightClippingSGDLearner, WeightClippingAdamLearner
from core.learner.sl.upgd import UPGDLearner, AdaUPGDLearner
from core.network.fcn_relu import FCNReLU
from core.network.fcn_tanh import FCNTanh
from core.network.fcn_leakyrelu import FCNLeakyReLU
from core.runner import Runner
from core.task.input_permuted_mnist import InputPermutedMNIST
from core.run.sl_run import SLRun
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "exp1"
task = InputPermutedMNIST()
total_steps = 15000000
n_seeds = 5

sgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.1, 0.01, 0.001, 0.0001],
               network=[FCNReLU(), FCNLeakyReLU()],
               n_samples=[total_steps],
    )

adam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01, 0.001, 0.0001, 0.00001],
               network=[FCNReLU(), FCNLeakyReLU()],
               n_samples=[total_steps],
    )

l2_init_sgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.1, 0.01, 0.001, 0.0001],
               weight_decay=[0.1, 0.01, 0.001, 0.0001],
               network=[FCNReLU(), FCNLeakyReLU()],
               n_samples=[total_steps],
    )

l2_init_adam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01, 0.001, 0.0001, 0.00001],
               weight_decay=[0.1, 0.01, 0.001, 0.0001],
               network=[FCNReLU(), FCNLeakyReLU()],
               n_samples=[total_steps],
    )

madam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.1, 0.01, 0.001, 0.0001],
               p_scale=[1.0, 2.0, 3.0, 4.0, 5.0],
               network=[FCNReLU(), FCNLeakyReLU()],
               n_samples=[total_steps],
    )

shrink_and_perturb_sgd_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.1, 0.01, 0.001, 0.0001],
                sigma=[0.0, 0.1, 0.01, 0.001],
                weight_decay=[0.0, 0.1, 0.01, 0.001],
               network=[FCNReLU(), FCNLeakyReLU()],
                n_samples=[total_steps],
     )

shrink_and_perturb_adam_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.01, 0.001, 0.0001, 0.00001],
                sigma=[0.0, 0.1, 0.01, 0.001],
                weight_decay=[0.0, 0.1, 0.01, 0.001],
               network=[FCNReLU(), FCNLeakyReLU()],
                n_samples=[total_steps],
     )

weight_clipping_sgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.1, 0.01, 0.001, 0.0001],
               zeta=[1.0, 2.0, 3.0, 4.0, 5.0],
               clip_last_layer=[0, 1],
               network=[FCNReLU(), FCNLeakyReLU()],
               n_samples=[total_steps],
    )


weight_clipping_adam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01, 0.001, 0.0001, 0.00001],
               zeta=[1.0, 2.0, 3.0, 4.0, 5.0],
               clip_last_layer=[0, 1],
               network=[FCNReLU(), FCNLeakyReLU()],
               n_samples=[total_steps],
    )


upgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.1, 0.01, 0.001, 0.0001],
               beta_utility=[0.99, 0.999, 0.9999],
               weight_decay=[0.1, 0.01, 0.001, 0.0001],
               sigma=[0.0, 0.1, 0.01, 0.001],
               network=[FCNReLU(), FCNLeakyReLU()],
               n_samples=[total_steps],
    )


adaupgd_adam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01, 0.001, 0.0001, 0.00001],
               beta_utility=[0.9, 0.99, 0.999, 0.9999],
               weight_decay=[0.1, 0.01, 0.001, 0.0001],
               sigma=[0.0, 0.1, 0.01, 0.001],
               network=[FCNReLU(), FCNLeakyReLU()],
               n_samples=[total_steps],
    )


grids = [
        sgd_grid,
        adam_grid,
        l2_init_sgd_grid,
        l2_init_adam_grid,
        madam_grid,
        shrink_and_perturb_sgd_grid,
        shrink_and_perturb_adam_grid,
        weight_clipping_sgd_grid,
        weight_clipping_adam_grid,
        # upgd_grid,
        # adaupgd_adam_grid,
]

learners = [
    SGDLearner(),
    AdamLearner(),
    L2InitSGDLearner(),
    L2InitAdamLearner(),
    MadamLearner(),
    ShrinkAndPerturbSGDLearner(),
    ShrinkAndPerturbAdamLearner(),
    WeightClippingSGDLearner(),
    WeightClippingAdamLearner(),
    # UPGDLearner(),
    # AdaUPGDLearner(),
]

save_dir = "generated_cmds"
for learner, grid in zip(learners, grids):
    runner = Runner(SLRun(), learner, task, grid, exp_name)
    num_jobs = runner.write_cmd(save_dir)
    create_script_generator(save_dir, exp_name, learner.name, num_jobs, time="60:00:00", memory="1G")
create_script_runner(save_dir, exp_name)