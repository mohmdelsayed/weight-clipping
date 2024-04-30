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

    def plot(self, save_name="accuracy"):
        # WC, None, L2 Init, S&P, Madam
        # colors = ["tab:blue", "tab:red", "tab:orange", "tab:green", "tab:purple"]
        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    configuration_list.append(data[self.what_to_plot])
                    learner_name = data["learner"]
            if learner_name == "sgd":
                color = "tab:red"
                linestyle = "--"
            elif learner_name == "adam":
                color = "tab:red"
                linestyle = "-"
            elif learner_name == "weight_clipping_sgd":
                color = "tab:blue"
                linestyle = "--"
            elif learner_name == "weight_clipping_adam":
                color = "tab:blue"
                linestyle = "-"
            elif learner_name == "l2_init_sgd":
                color = "tab:orange"
                linestyle = "--"
            elif learner_name == "l2_init_adam":
                color = "tab:orange"
                linestyle = "-"
            elif learner_name == "shrink_and_perturb_sgd":
                color = "tab:green"
                linestyle = "--"
            elif learner_name == "shrink_and_perturb_adam":
                color = "tab:green"
                linestyle = "-"
            elif learner_name == "madam":
                color = "tab:purple"
                linestyle = "-"
            else:
                raise("Error")

            configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)
            mean_list = np.array(configuration_list).mean(axis=0)
            std_list = np.array(configuration_list).std(axis=0) / np.sqrt(len(seeds))
            x_s = [2*i for i in range(len(mean_list))]
            plt.plot(x_s, mean_list, label=learner_name, linewidth=2, color=color, linestyle=linestyle)
            plt.fill_between(x_s, mean_list - std_list, mean_list + std_list, alpha=0.2, color=color)
        
        if self.what_to_plot == "losses":
            plt.ylabel("Average Online Loss", fontsize=26)
        elif self.what_to_plot == "accuracies":
            plt.ylabel("Average Online Accuracy", fontsize=26)
        elif self.what_to_plot == "plasticity_per_task":
            plt.ylabel("Average Online Plasticity", fontsize=26)
        elif self.what_to_plot == "clipping_proportion":
            plt.ylabel("\% Weight Clipping", fontsize=26)
        elif self.what_to_plot == "n_dead_units_per_task":
            plt.ylabel("\% Zero Activations", fontsize=26)
        elif self.what_to_plot == "weight_rank_per_task":
            plt.ylabel("Weight Rank", fontsize=26)
        elif self.what_to_plot == "weight_l2_per_task":
            plt.ylabel(r"$\ell_2$ Norm of Weights", fontsize=26)
        elif self.what_to_plot == "weight_l1_per_task":
            plt.ylabel(r"$\ell_1$ Norm of Weights", fontsize=26)
        elif self.what_to_plot == "grad_l2_per_task":
            plt.ylabel(r"$\ell_2$ Norm of Gradients", fontsize=26)
        elif self.what_to_plot == "grad_l1_per_task":
            plt.ylabel(r"$\ell_1$ Norm of Gradients", fontsize=26)
        elif self.what_to_plot == "grad_l0_per_task":
            plt.ylabel(r"$\ell_0$ Norm of Gradients", fontsize=26)
        else:
            raise("error")
        # plt.ylim([0.71, 0.8])
        # plt.ylim(top=0.9)
        plt.xlabel(f"Task Number", fontsize=26)
        # plt.legend()
        # plt.title(self.task_name)
        plt.savefig(f"{save_name}.pdf", bbox_inches='tight')
        plt.clf()

if __name__ == "__main__":
    what_to_plot = "accuracies"
    # what_to_plot = "losses"
    # what_to_plot = "plasticity_per_task"
    # what_to_plot = "clipping_proportion"
    # what_to_plot = "n_dead_units_per_task"
    # what_to_plot = "weight_rank_per_task"
    # what_to_plot = "weight_l2_per_task"
    # what_to_plot = "weight_l1_per_task"
    # what_to_plot = "grad_l2_per_task"
    # what_to_plot = "grad_l0_per_task"
    # what_to_plot = "grad_l1_per_task"
    
    best_runs = BestConfig("exp1_stats_1M/input_permuted_mnist_5k", "fcn_leakyrelu_with_hooks",  ["sgd", "adam", "madam", "l2_init_sgd", "l2_init_adam", "shrink_and_perturb_adam", "shrink_and_perturb_sgd", "weight_clipping_sgd", "weight_clipping_adam"]).get_best_run(measure="accuracies")
    # best_runs = BestConfig("exp2_stats/label_permuted_emnist", "fcn_leakyrelu_with_hooks",  ["sgd", "adam", "madam", "l2_init_sgd", "l2_init_adam", "shrink_and_perturb_adam", "shrink_and_perturb_sgd", "weight_clipping_sgd", "weight_clipping_adam"]).get_best_run(measure="accuracies")
    # best_runs = BestConfig("exp3_stats/label_permuted_mini_imagenet", "fcn_leakyrelu_with_hooks",  ["sgd", "adam", "madam", "l2_init_sgd", "l2_init_adam", "shrink_and_perturb_adam", "shrink_and_perturb_sgd", "weight_clipping_sgd", "weight_clipping_adam"]).get_best_run(measure="accuracies")

    print(best_runs)
    plotter = SLPlotter(best_runs, task_name="Label-Permuted mini-ImageNet", avg_interval=2, what_to_plot=what_to_plot)
    plotter.plot(save_name=what_to_plot)
