import os
import numpy as np
import pandas as pd
import argparse
from graph_desciptors import calc_graph_descriptors
from joblib import Parallel, delayed
from time import time
from datetime import datetime

# Setting path------------------------------------------------------------------
dir_path = os.getcwd()

descriptors_dir = os.path.join(dir_path,
                            "data/to_kanamori/cohesive/descriptors")
compounds_list_cohesive = os.path.join(dir_path,
                            "data/to_kanamori/cohesive/compounds_name")
compounds_list_ltc = os.path.join(dir_path, "data/to_kanamori/ltc/kappa")
# compounds_list_mp = os.path.join(dir_path,
#                             "data/to_kanamori/melting_temp/mp_data")
compounds_list_mp = os.path.join(dir_path,
                            "data/to_kanamori/melting_temp/mp_data_no_simple")
# ------------------------------------------------------------------------------

atomic_df = pd.read_csv(dir_path + "/data/atomic_data_reduced.csv", index_col=0)
atomic_df = atomic_df.drop(["kai(Allen)", "Rps-d"], axis=1)

parser = argparse.ArgumentParser(description="compute graph descriptors")
parser.add_argument("--property", required=True, help="property of a dataset")
parser.add_argument("--save_dir", required=True, help="dir_path to save data")
parser.add_argument("--isTest", action="store_true", help="whether to test or not")
args = parser.parse_args()

load_dir = os.path.join('data/descriptors', args.property)
# adj_matrix_no_weight = np.load(os.path.join(load_dir, 'adj_matrix_no_weight.npy'))
# adj_matrix_multi_edge = np.load(os.path.join(load_dir, 'adj_matrix_multi_edge.npy'))
# adj_matrix_sol_angle = np.load(os.path.join(load_dir, 'adj_matrix_sol_angle.npy'))


def save_df(descriptor, compounds_list, name):
    df_descriptor = pd.DataFrame(descriptor,
                            index=compounds_list,
                            columns=range(len(descriptor[0])))
    df_descriptor.to_csv(os.path.join(args.save_dir, name+'.csv'))


if __name__ == "__main__":

    if args.property == "cohesive":
        compounds_list_path = compounds_list_cohesive
    elif args.property == "ltc":
        compounds_list_path = compounds_list_ltc
        atomic_df = atomic_df.drop(["IE2", "Rps-s"], axis=1)
    elif args.property == "mp":
        compounds_list_path = compounds_list_mp
        atomic_df = atomic_df.drop(["IE2", "Rps-s"], axis=1)
    else:
        raise ValueError('please choose a valid property name')

    start = time()
    print('Started at {}'.format(datetime.now()))

    with open(compounds_list_path) as f:
        lines = f.readlines()
        if args.isTest == True:
            n_samples = 3
        else:
            n_samples = len(lines)

        simple_sum = []
        simple_mean = []
        simple_std = []
        multi_edge_sum = []
        multi_edge_mean = []
        multi_edge_std = []
        sol_angle_sum = []
        sol_angle_mean = []
        sol_angle_std = []
        compounds_list = []
        for index in range(n_samples):
            if args.property == "cohesive":
                compound_dir = lines[index].strip()
                POSCAR_path = os.path.join(descriptors_dir, compound_dir, "POSCAR")
            elif args.property == "ltc" or args.property == "mp":
                line = lines[index].strip().split()
                compound_dir = line[0]
                POSCAR_path = os.path.join(dir_path, compound_dir, "POSCAR")

            if index % 10 == 0:
                print(index)
            # adj_matrix_list = [adj_matrix_no_weight[index],
            #                     adj_matrix_multi_edge[index],
            #                     adj_matrix_sol_angle[index]]
            # compute descriptors
            descriptor = calc_graph_descriptors(atomic_df, POSCAR_path,
                                                adj_matrix_list=None)
            simple_sum.append(descriptor[0])
            simple_mean.append(descriptor[1])
            simple_std.append(descriptor[2])
            multi_edge_sum.append(descriptor[3])
            multi_edge_mean.append(descriptor[4])
            multi_edge_std.append(descriptor[5])
            sol_angle_sum.append(descriptor[6])
            sol_angle_mean.append(descriptor[7])
            sol_angle_std.append(descriptor[8])

            compounds_list.append(compound_dir)

        print('It took {} sec.'.format(time() - start))

        save_df(simple_sum, compounds_list, 'simple_sum')
        save_df(simple_mean, compounds_list, 'simple_mean')
        save_df(simple_std, compounds_list, 'simple_std')
        save_df(multi_edge_sum, compounds_list, 'multi_edge_sum')
        save_df(multi_edge_mean, compounds_list, 'multi_edge_mean')
        save_df(multi_edge_std, compounds_list, 'multi_edge_std')
        save_df(sol_angle_sum, compounds_list, 'sol_angle_sum')
        save_df(sol_angle_mean, compounds_list, 'sol_angle_mean')
        save_df(sol_angle_std, compounds_list, 'sol_angle_std')
