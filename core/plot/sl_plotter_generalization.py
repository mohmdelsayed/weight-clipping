import json
import matplotlib.pyplot as plt
from core.best_config import BestConfig
import os
import numpy as np
import matplotlib
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 14})

class SLPlotter:
    def __init__(self, best_runs_path, task_name, avg_interval=1, what_to_plot="losses"):
        self.best_runs_path = best_runs_path
        self.avg_interval = avg_interval
        self.task_name = task_name
        self.what_to_plot = what_to_plot

    def plot(self):
        colors = ['tab:blue', 'tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown']
        # names = ['SGD+WC (50\%)', 'SGD (50\%)', 'SGD (100\%)', 'a','b', 'c', 'd']
        names = ['1','2','3']
        # names = ['SGD+WC@300 (50\%)', 'SGD+WC (50\%)', 'SGD (50\%)', 'SGD (100\%)', 'a','b', 'c', 'd']

        # names = ['SGD+WC', 'SGD']
        for name, color, subdir in zip(names, colors, self.best_runs_path):
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    configuration_list.append(data[self.what_to_plot])

            configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)
            mean_list = np.array(configuration_list).mean(axis=0)
            std_list = np.array(configuration_list).std(axis=0) / np.sqrt(len(seeds))
            if 'exp4' in subdir:
                xs = range(len(mean_list), 2*len(mean_list), 1)
            else:
                xs = range(len(mean_list))

            # xs = range(len(mean_list))
            if 'SGD+WC@300 (50\%)' in name:
                lim = 299
                plt.plot(xs[lim:], mean_list[lim:], label=name, color=color, linestyle='dotted')
                plt.fill_between(xs[lim:], mean_list[lim:] - std_list[lim:], mean_list[lim:] + std_list[lim:], alpha=0.2, color=color)
            else:
                plt.plot(xs, mean_list, label=name, color=color)
                plt.fill_between(xs, mean_list - std_list, mean_list + std_list, alpha=0.2, color=color)
            if self.what_to_plot == "losses":
                plt.ylabel("Online Loss")
            else:
                plt.ylabel("Average Test Accuracy", fontsize=26)
        plt.legend()
        # plt.axvline(x=300, ls='--', color='black', linewidth=0.8)
        plt.xlabel(f"Epoch Number", fontsize=26)
        # plt.ylim(bottom=0.2)
        plt.savefig("plot.pdf", bbox_inches='tight')
        plt.clf()

if __name__ == "__main__":
    what_to_plot = "accuracies"
    best_runs = BestConfig("exp4/cifar10", "resnet18",  ["weight_clipping_adam", "adam"]).get_best_run(measure=what_to_plot)
    # print(best_runs)

    # best_runs = [
    #     'logs/exp5/cifar10_nonstationary/weight_clipping_sgd/resnet18/lr_0.001_zeta_20.0',
    #     'logs/exp5/cifar10_nonstationary/sgd/resnet18/lr_0.001',
    #     'logs/exp4/cifar10/sgd/resnet18/lr_0.001',
    #     # 'logs/exp4/cifar10/weight_clipping_sgd/resnet18/lr_0.001_zeta_20.0',
    # ]

    # best_runs = [
    #     # 'logs/exp6_zeta5/cifar10_nonstationary/sgd/resnet18/lr_0.001',
    #     'logs/exp6_zeta10/cifar10_nonstationary/sgd/resnet18/lr_0.001',
    #     'logs/exp5/cifar10_nonstationary/weight_clipping_sgd/resnet18/lr_0.001_zeta_20.0',

    #     'logs/exp5/cifar10_nonstationary/sgd/resnet18/lr_0.001',
    #     'logs/exp4/cifar10/sgd/resnet18/lr_0.001',
    # ]


    # best_runs = [
    #     'logs/exp4/cifar10/weight_clipping_sgd/resnet18/lr_0.001_zeta_20.0',
    #     'logs/exp4/cifar10/sgd/resnet18/lr_0.001',
    # ]

    # best_runs = BestConfig("exp1_old/input_permuted_mnist", "fcn_relu",  ["upgd", "adam"]).get_best_run(measure=what_to_plot)
    # print(best_runs)
    # best_runs = ['logs/exp1_old/input_permuted_mnist/upgd/fcn_relu/lr_0.001_beta_utility_0.9999_weight_decay_0.001_sigma_0.1', 'logs/exp1_old/input_permuted_mnist/adam/fcn_relu/lr_0.0001']
    plotter = SLPlotter(best_runs, task_name="CIFAR-10", avg_interval=2, what_to_plot=what_to_plot)
    plotter.plot()