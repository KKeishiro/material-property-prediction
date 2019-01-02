import os
import numpy as np
import pandas as pd
import argparse
from laplacian import set_structure_from_poscar_file, get_adj_matrix
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
parser = argparse.ArgumentParser(description="compute adjacency matrix")
parser.add_argument("--property", required=True, help="property of a dataset")
parser.add_argument("--tol", default=0, help="tolerance parameter")
parser.add_argument("--cutoff", default=10, type=float, help="cutoff value")
parser.add_argument("--isTest", action="store_true", help="whether to test or not")
args = parser.parse_args()

save_dir = os.path.join('data/descriptors', args.property)

def main(index, property):
    if property == "cohesive":
        compound_dir = lines[index].strip()
        POSCAR_path = os.path.join(descriptors_dir, compound_dir, "POSCAR")
    elif property == "ltc" or property == "mp":
        line = lines[index].strip().split()
        compound_dir = line[0]
        POSCAR_path = os.path.join(dir_path, compound_dir, "POSCAR")

    # compute descriptors
    structure = set_structure_from_poscar_file(POSCAR_path)
    primitive_structure = structure.get_primitive_structure()
    adj_matrix_no_weight, adj_matrix_multi_edge, adj_matrix_sol_angle = \
            get_adj_matrix(primitive_structure, tol=args.tol, cutoff=args.cutoff)
    return adj_matrix_no_weight, adj_matrix_multi_edge, adj_matrix_sol_angle, \
            compound_dir


if __name__ == "__main__":

    if args.property == "cohesive":
        compounds_list_path = compounds_list_cohesive
    elif args.property == "ltc":
        compounds_list_path = compounds_list_ltc
    elif args.property == "mp":
        compounds_list_path = compounds_list_mp
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

        results = Parallel(n_jobs=-1, verbose=2)(
              delayed(main)(index, args.property) for index in range(n_samples))
        results = np.array(results)
        adj_matrix_no_weight = results[:,0]
        adj_matrix_multi_edge = results[:,1]
        adj_matrix_sol_angle = results[:,2]
        compounds_list = results[:,3]

        print('It took {} sec.'.format(time() - start))

        np.save(os.path.join(save_dir, 'adj_matrix_no_weight.npy'), adj_matrix_no_weight)
        np.save(os.path.join(save_dir, 'adj_matrix_multi_edge.npy'), adj_matrix_multi_edge)
        np.save(os.path.join(save_dir, 'adj_matrix_sol_angle.npy'), adj_matrix_sol_angle)
        np.save(os.path.join(save_dir, 'compounds_list.npy'), compounds_list)
