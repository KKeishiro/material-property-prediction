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
compounds_list_mp = os.path.join(dir_path,
                            "data/to_kanamori/melting_temp/mp_data")
# ------------------------------------------------------------------------------

atomic_df = pd.read_csv(dir_path + "/data/atomic_data_reduced.csv", index_col=0)
atomic_df = atomic_df.drop(["Rps-d"], axis=1)

parser = argparse.ArgumentParser(description="compute laplacian-based descriptors")
parser.add_argument("--property", required=True, help="property of a dataset")
parser.add_argument("--tol", default=0, help="tolerance parameter")
parser.add_argument("--cutoff", default=10, type=float, help="cutoff value")
parser.add_argument("--save_path", required=True, help="path to save data")
parser.add_argument("--isTest", action="store_true", help="whether to test or not")
args = parser.parse_args()


def main(index, property):
    if property == "cohesive":
        compound_dir = lines[index].strip()
        POSCAR_path = os.path.join(descriptors_dir, compound_dir, "POSCAR")
    elif property == "ltc" or property == "mp":
        line = lines[index].strip().split()
        compound_dir = line[0]
        POSCAR_path = os.path.join(dir_path, compound_dir, "POSCAR")

    # compute descriptors
    descriptor = calc_laplacian_descriptors(atomic_df, POSCAR_path,
                                tol=args.tol,
                                cutoff=args.cutoff)
    return descriptor, compound_dir


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
            n_samples = 5
        else:
            n_samples = len(lines)

        results = Parallel(n_jobs=-1, verbose=2)(
              delayed(main)(index, args.property) for index in range(n_samples))
        results = np.array(results)
        descriptors = results[:,0]
        compounds_list = results[:,1]

        print('It took {} sec.'.format(time() - start))

        df_descriptors = pd.DataFrame(descriptors, index=compounds_list)
        df_descriptors.to_csv(args.save_path)
