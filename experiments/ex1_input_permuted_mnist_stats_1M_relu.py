from core.grid_search import GridSearch
from core.learner.sl.sgd import SGDLearner
from core.learner.sl.adam import AdamLearner
from core.learner.sl.l2_init import L2InitSGDLearner, L2InitAdamLearner
from core.learner.sl.madam import MadamLearner
from core.learner.sl.shrink_and_pertub import ShrinkAndPerturbSGDLearner, ShrinkAndPerturbAdamLearner
from core.learner.sl.weight_clipping import WeightClippingSGDLearner, WeightClippingAdamLearner
from core.learner.sl.upgd import UPGDLearner, AdaUPGDLearner
from core.network.fcn_relu import FCNReLUWithHooks as FCNReLU
from core.network.fcn_tanh import FCNTanhWithHooks as FCNTanh
from core.network.fcn_leakyrelu import FCNLeakyReLUWithHooks as FCNLeakyReLU
from core.runner import Runner
from core.task.input_permuted_mnist import InputPermutedMNIST
from core.run.sl_stats import RunStats
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "exp1_stats_1M"
task = InputPermutedMNIST()
total_steps = 1000000
n_seeds = 20

# 'logs/exp1/input_permuted_mnist1M/sgd/fcn_relu/lr_0.001',
sgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               network=[FCNReLU()],
               n_samples=[total_steps],
    )

# 'logs/exp1/input_permuted_mnist1M/adam/fcn_relu/lr_0.0001',
adam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.0001],
               network=[FCNReLU()],
               n_samples=[total_steps],
    )

# 'logs/exp1/input_permuted_mnist1M/l2_init_sgd/fcn_relu/lr_0.001_weight_decay_0.01',
l2_init_sgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               weight_decay=[0.01],
               network=[FCNReLU()],
               n_samples=[total_steps],
    )

# 'logs/exp1/input_permuted_mnist1M/l2_init_adam/fcn_relu/lr_0.0001_weight_decay_0.01',
l2_init_adam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.0001],
               weight_decay=[0.01],
               network=[FCNReLU()],
               n_samples=[total_steps],
    )

# 'logs/exp1/input_permuted_mnist1M/madam/fcn_relu/lr_0.001_p_scale_5.0',
madam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               p_scale=[5.0],
               network=[FCNReLU()],
               n_samples=[total_steps],
    )


# 'logs/exp1/input_permuted_mnist1M/shrink_and_perturb_sgd/fcn_relu/lr_0.001_sigma_0.1_weight_decay_0.01',
shrink_and_perturb_sgd_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.001],
                sigma=[0.1],
                weight_decay=[0.01],
               network=[FCNReLU()],
                n_samples=[total_steps],
     )

# 'logs/exp1/input_permuted_mnist1M/shrink_and_perturb_adam/fcn_relu/lr_0.0001_sigma_0.1_weight_decay_0.01',
shrink_and_perturb_adam_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.0001],
                sigma=[0.1],
                weight_decay=[0.01],
               network=[FCNReLU()],
                n_samples=[total_steps],
     )

# 'logs/exp1/input_permuted_mnist1M/weight_clipping_sgd/fcn_relu/lr_0.001_zeta_2.0',
# 'logs/exp1/input_permuted_mnist1M/weight_clipping_sgd/fcn_relu/lr_0.001_zeta_1.0',#
weight_clipping_sgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               zeta=[1.0, 2.0],
               network=[FCNReLU()],
               n_samples=[total_steps],
    )

# 'logs/exp1/input_permuted_mnist1M/weight_clipping_adam/fcn_relu/lr_0.0001_zeta_1.0'
weight_clipping_adam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.0001],
               zeta=[1.0],
               network=[FCNReLU()],
               n_samples=[total_steps],
    )


# UPGD: α = 0.01,σ = 0.1, βu = 0.9999,λ = 0.01
upgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01],
               beta_utility=[0.9999],
               weight_decay=[0.01],
               sigma=[0.1],
               network=[FCNReLU()],
               n_samples=[total_steps],
    )


# adaupgd_adam_grid = GridSearch(
#                seed=[i for i in range(0, n_seeds)],
#                lr=[0.01, 0.001, 0.0001, 0.00001],
#                beta_utility=[0.99, 0.999, 0.9999],
#                weight_decay=[0.1, 0.01, 0.001, 0.0001],
#                sigma=[0.0, 0.1, 0.01, 0.001],
#                network=[FCNReLU()],
#                n_samples=[total_steps],
#     )


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
        upgd_grid,
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
    UPGDLearner(),
    # AdaUPGDLearner(),
]

save_dir = "generated_cmds"
for learner, grid in zip(learners, grids):
    runner = Runner(RunStats(), learner, task, grid, exp_name)
    num_jobs = runner.write_cmd(save_dir)
    create_script_generator(save_dir, exp_name, learner.name, num_jobs, time="80:00:00", memory="3G")
create_script_runner(save_dir, exp_name)