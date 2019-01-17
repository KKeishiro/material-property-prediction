import os
import numpy as np
import pandas as pd
import argparse
from laplacian import calc_laplacian_descriptors
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

parser = argparse.ArgumentParser(description="compute laplacian-based descriptors")
parser.add_argument("--property", required=True, help="property of a dataset")
parser.add_argument("--save_path", required=True, help="path to save data")
parser.add_argument("--isTest", action="store_true", help="whether to test or not")
parser.add_argument("--potential", action="store_true",
                    help="potential is taken into account")
parser.add_argument("--oxi_state_guesses", action="store_true",
                    help="oxi_state will be guessed")
args = parser.parse_args()

load_dir = os.path.join('data/descriptors', args.property)
adj_matrix_no_weight = np.load(os.path.join(load_dir, 'adj_matrix_no_weight.npy'))
adj_matrix_multi_edge = np.load(os.path.join(load_dir, 'adj_matrix_multi_edge.npy'))
adj_matrix_sol_angle = np.load(os.path.join(load_dir, 'adj_matrix_sol_angle.npy'))

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
        assert False, 'please choose a valid property name'

    start = time()
    print('Started at {}'.format(datetime.now()))

    with open(compounds_list_path) as f:
        lines = f.readlines()
        if args.isTest == True:
            n_samples = 20
        else:
            n_samples = len(lines)

        descriptors = []
        compounds_list = []
        for index in range(n_samples):
            if args.property == "cohesive":
                compound_dir = lines[index].strip()
                POSCAR_path = os.path.join(descriptors_dir, compound_dir, "POSCAR")
            elif args.property == "ltc" or args.property == "mp":
                line = lines[index].strip().split()
                compound_dir = line[0]
                POSCAR_path = os.path.join(dir_path, compound_dir, "POSCAR")

            if index % 1000 == 0:
                print(index)
            adj_matrix_list = [adj_matrix_no_weight[index],
                                adj_matrix_multi_edge[index],
                                adj_matrix_sol_angle[index]]
            # compute descriptors
            descriptor = calc_laplacian_descriptors(atomic_df, POSCAR_path,
                                    adj_matrix_list,
                                    potential=args.potential,
                                    oxi_state_guesses=args.oxi_state_guesses)
            descriptors.append(descriptor)
            compounds_list.append(compound_dir)

        print('It took {} sec.'.format(time() - start))

        df_descriptors = pd.DataFrame(descriptors,
                                index=compounds_list,
                                columns=range(len(descriptors[0])))
        df_descriptors.to_csv(args.save_path)
