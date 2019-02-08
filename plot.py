import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
sns.set()
sns.set_style('whitegrid')

def get_laplacian(adj_matrix):
    deg_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian = deg_matrix - adj_matrix
    return laplacian


cohesive_res = {'simple': [0.200,0.126,0.143],
                'multi_edge': [0.193,0.144,0.142],
                'sol_angle': [0.194,0.110,0.123],
                'all': [0.155, 0.075, 0.073],
                'laplacian': [0.064,0.053]}

ltc_res = {'simple': [0.20,0.15,0.13],
                'multi_edge': [0.20,0.15,0.14],
                'sol_angle': [0.20,0.17,0.16],
                'all': [0.22, 0.15, 0.13],
                'laplacian': [0.12,0.11]}

mp_res = {'simple': [310,290,280],
                'multi_edge': [320,270,280],
                'sol_angle': [300,280,280],
                'all': [320, 300, 280],
                'laplacian': [290,280]}

# plot
def plot(property='cohesive', combine=False):
    if property == 'cohesive':
        res = cohesive_res
        title = 'Cohesive energy'
        ylabel = 'RMSE (eV/atom)'
        ylim = 0, 0.32
    elif property == 'ltc':
        res = ltc_res
        title = 'LTC'
        ylabel = 'RMSE (W/mK)'
        ylim = 0, 0.25
    else: # mp
        res = mp_res
        title = 'Melting temperature'
        ylabel = 'RMSE (K)'
        ylim = 0, 400

    sns.set_palette('husl')
    x = np.array(['sum', 'mean', 'std'])
    x_position = np.arange(len(x))

    fig, ax = plt.subplots()

    if combine:
        ax.bar(x_position + 0.2, res['all'], width=0.2, color='b')
    else:
        ax.bar(x_position, res['simple'], width=0.2, label='simple')
        ax.bar(x_position + 0.2, res['multi_edge'], width=0.2, label='multi_edge')
        ax.bar(x_position + 0.4, res['sol_angle'], width=0.2, label='sol_angle')
        ax.legend(loc='upper right', fontsize='small')
    ax.set_xticks(x_position + 0.2)
    ax.set_xticklabels(x)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.2, 2.6)
    ax.set_ylim(ylim)
    if property == 'cohesive':
        ax.plot([-0.5,3], [0.243,0.243], color='k')
        ax.text(x=0.7, y=0.282,s='Without graph features (0.243)')
        ax.annotate('', xy=(1.2,0.245), xytext=(1.2,0.275), arrowprops=dict(color='k'))
    elif property == 'ltc':
        ax.plot([-0.5,3], [0.17,0.17], color='k')
        ax.text(x=0.7, y=0.21,s='Without graph features (0.17)')
        ax.annotate('', xy=(1.2,0.175), xytext=(1.2,0.205), arrowprops=dict(color='k'))
    else:
        ax.plot([-0.5,3], [300,300], color='k')
        ax.text(x=0.7, y=355,s='Without graph features (300)')
        ax.annotate('', xy=(1.2,305), xytext=(1.2,340), arrowprops=dict(color='k'))
    plt.show()

# plot graph laplacian
def plot_laplacian(property='cohesive'):
    if property == 'cohesive':
        res = cohesive_res
        title = 'Cohesive energy'
        ylabel = 'RMSE (eV/atom)'
        ylim = 0, 0.32
    elif property == 'ltc':
        res = ltc_res
        title = 'LTC'
        ylabel = 'RMSE (W/mK)'
        ylim = 0, 0.25
    else: # mp
        res = mp_res
        title = 'Melting temperature'
        ylabel = 'RMSE (K)'
        ylim = 0, 400

    x = np.array(['w/o potential', 'with potential'])
    x_position = np.arange(len(x))

    fig, ax = plt.subplots()
    ax.bar(x_position, res['laplacian'], width=0.3, color='b')
    ax.set_xticks(x_position)
    ax.set_xticklabels(x)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(ylim)
    if property == 'cohesive':
        ax.plot([-0.5,3], [0.243,0.243], color='k')
        ax.plot([-0.5,3], [0.073,0.073], color='r')
        ax.text(x=0.2, y=0.282,s='Without graph features (0.243)')
        ax.text(x=0.17, y=0.12,s='Best model by std features (0.073)', color='r')
        ax.annotate('', xy=(0.5,0.245), xytext=(0.5,0.275), arrowprops=dict(color='k'))
        ax.annotate('', xy=(0.5,0.08), xytext=(0.5,0.11), arrowprops=dict(color='r'))
    elif property == 'ltc':
        ax.plot([-0.5,3], [0.17,0.17], color='k')
        ax.plot([-0.5,3], [0.13,0.13], color='r')
        ax.text(x=0.2, y=0.21,s='Without graph features (0.17)')
        ax.text(x=0.17, y=0.155,s='Best model by std features (0.13)', color='r')
        ax.annotate('', xy=(0.5,0.175), xytext=(0.5,0.205), arrowprops=dict(color='k'))
        ax.annotate('', xy=(0.5,0.13), xytext=(0.5,0.15), arrowprops=dict(color='r'))
    else:
        ax.plot([-0.5,3], [300,300], color='k')
        ax.plot([-0.5,3], [270,270], color='r')
        ax.text(x=-0.1, y=355,s='Without graph features (300)')
        ax.text(x=0.3, y=320,s='Best model by multi_edge mean features (270)',
                color='r')
        ax.annotate('', xy=(0.15,305), xytext=(0.15,340), arrowprops=dict(color='k'))
        ax.annotate('', xy=(0.6,275), xytext=(0.6,315), arrowprops=dict(color='r'))
    plt.show()


features = np.load('data/feature_importance/feature_name.npy')
perm_importance = np.load('data/feature_importance/perm_importance.npy')
std_perm_importance = np.load('data/feature_importance/perm_importance_std.npy')
def plot_perm_importance(perm_importance, std_perm_importance):
    ranking = np.argsort(-perm_importance)
    fig, ax = plt.subplots(figsize=(9, 11))
    sns.barplot(x=perm_importance[ranking[:30]], y=features[ranking[:30]],
                orient='h')
    ax.set_title("Permutation importance (top30)")
    ax.set_xlabel("Increase in RMSE (eV / atom)")
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--property", required=True, help="property")
    parser.add_argument("--laplacian", action="store_true",
                        help="whether plot laplacian result")
    parser.add_argument("--combined", action="store_true",
                        help="plot the result for combined features")
    parser.add_argument("--perm_importance", action='store_true',
                        help="plot permutation importance")
    args = parser.parse_args()
    # if args.laplacian:
    #     plot_laplacian(args.property)
    # else:
    #     plot(args.property, args.combined)

    if args.perm_importance:
        plot_perm_importance(perm_importance, std_perm_importance)
