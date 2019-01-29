import os.path as osp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import seaborn as sns
sns.set_style('whitegrid')

descriptors_dir = 'data/descriptors'
parser = argparse.ArgumentParser(description='visualization')
parser.add_argument('--property', help='property of a dataset')
parser.add_argument('--predicted', help='predction result file')
args = parser.parse_args()


def plot_distribution(y):
    f,ax1 = plt.subplots()
    sns.distplot(y['mp'], bins=20, kde=False, rug=True, ax=ax1)
    ax2 = ax1.twinx()
    ax2.set_ylim(0,0.001)
    ax2.yaxis.set_ticks([])
    sns.kdeplot(y['mp'], ax=ax2, legend=False)
    ax1.set_xlabel('Melting temperature [K]')
    ax1.set_ylabel('Counts')
    plt.show()


def plot_prediction_result(y, predicted):
    f,ax =plt.subplots(figsize=(5.5,10))
    ax.scatter(y, predicted, s=20, alpha=0.4, color='b')
    ax.plot(np.array(range(9)), np.array(range(9)), ls='--', lw=1, color='k')
    ax.set_xlabel("DFT cohesive energy (eV / atom)", fontsize=14)
    ax.set_ylabel("Predicted cohesive energy (eV / atom)", fontsize=14)
    ax.set_xlim([0, 7.5])
    ax.set_ylim([0, 7.5])
    plt.show()


def plot_learning_curves(val_errors_mean1, val_errors_mean2,
                        val_errors_std1, val_errors_std2):
    # this is hard coding, which is a damn way
    training_set_size = [1628, 3256, 4884, 6512, 8140, 9768, 11396, 13024,
                        14652, 16280]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(training_set_size, val_errors_mean1, color='g', linestyle='solid',
            marker='o', markersize=8, fillstyle='none', markerfacecolor='none',
            markeredgewidth=1, markeredgecolor='g', label="w/o potential")
    ax.plot(training_set_size, val_errors_mean2, color='m', linestyle='solid',
            marker='o', markersize=8, fillstyle='none', markerfacecolor='none',
            markeredgewidth=1, markeredgecolor='m', label="with potential")
    ax.errorbar(training_set_size, val_errors_mean1, yerr=val_errors_std1,
                capsize=4, capthick=1, ecolor='g', color='g', linewidth=1)
    ax.errorbar(training_set_size, val_errors_mean2, yerr=val_errors_std2,
                capsize=4, capthick=1, ecolor='m', color='m', linewidth=1)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_xlabel("Training set size", fontsize=12)
    ax.set_ylabel("RMSE [eV / atom]", fontsize=12)
    ax.axis([0, 18000, 0, 0.2])
    plt.show()

# # graph
# mean_rmse1= np.load(osp.join("figure/learning_curve", "sum_rmse_mean.npy"))
# std_rmse1 = np.load(osp.join("figure/learning_curve", "sum_rmse_std.npy"))
# mean_rmse2 = np.load(osp.join("figure/learning_curve", "std_rmse_mean.npy"))
# std_rmse2 = np.load(osp.join("figure/learning_curve", "std_rmse_std.npy"))

# laplacian
mean_rmse1= np.load(osp.join("figure/learning_curve",
                            "laplacian_no_potential_rmse_mean.npy"))
std_rmse1 = np.load(osp.join("figure/learning_curve",
                            "laplacian_no_potential_rmse_std.npy"))
mean_rmse2 = np.load(osp.join("figure/learning_curve",
                            "laplacian_potential_rmse_mean.npy"))
std_rmse2 = np.load(osp.join("figure/learning_curve",
                            "laplacian_potential_rmse_std.npy"))


if __name__=='__main__':

    if args.property == 'cohesive':
        y = pd.read_csv(osp.join("data/to_kanamori",
                        "cohesive/coh_energy.csv"), header=None)
    elif args.property == 'ltc':
        y = pd.read_csv(osp.join("data/to_kanamori",
                        "ltc/kappa.csv"))
    elif args.property == 'mp':
        y = pd.read_csv(osp.join("data/to_kanamori",
                        "melting_temp/mp_data_no_simple.csv"))

    # predicted = np.load(args.predicted)
    # plot_prediction_result(y.values, predicted)

    plot_learning_curves(mean_rmse1, mean_rmse2, std_rmse1, std_rmse2)
