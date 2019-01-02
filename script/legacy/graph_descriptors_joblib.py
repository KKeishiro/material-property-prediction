import os
import numpy as np
import pandas as pd
import argparse
from data_processing import calculate_descriptors
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

parser = argparse.ArgumentParser(description="compute graph-based descriptors")
parser.add_argument("--property", required=True, help="property of a dataset")
parser.add_argument("--tol", default=1e-4, help="tolerance parameter")
parser.add_argument("--cutoff", default=10, type=float, help="cutoff value")
parser.add_argument("--cov_radii_tol", default=0.65,
                    help="Cordero covalent radii tolerance")
parser.add_argument("--connectivity", default="weight",
                    help="definition of connectivity")
parser.add_argument("--multiply_type", default="element_wise", help="multipication type")
parser.add_argument("--periodic", action="store_true",
                    help="whether periodic boundary condition is considered or not")
parser.add_argument("--descriptor_type", default="mixtured", help="descriptor type")
parser.add_argument("--max_matrix_power", default=10,
                    help="define how many descriptors will be obtained")
parser.add_argument("--quantize_strc", default="trace",
                    help="quantize type for structure descriptors")
parser.add_argument("--quantize_mix", default="sum",
                    help="quantize type for mixtured descriptors")
parser.add_argument("--save_path", required=True, help="path to save data")
parser.add_argument("--isTest", action="store_true", help="whether to test or not")
args = parser.parse_args()


def main(index, property):
    if property == "cohesive":
        compound_dir = lines[index].strip()
        POSCAR_path = descriptors_dir + compound_dir + "/POSCAR"
    elif property == "ltc" or property == "mp":
        line = lines[index].strip().split()
        compound_dir = line[0]
        POSCAR_path = dir_path + compound_dir + "/POSCAR"

    # compute descriptors
    descriptor = calculate_descriptors(atomic_df, POSCAR_path,
                                tol=args.tol,
                                cutoff=args.cutoff,
                                cov_radii_tol=args.cov_radii_tol,
                                connectivity=args.connectivity,
                                multiplication_type=args.multiply_type,
                                isPeriodic=args.periodic,
                                descriptor_type=args.descriptor_type,
                                max_matrix_power=args.max_matrix_power,
                                quantize_type_for_structure=args.quantize_strc,
                                quantize_type_for_mixtured=args.quantize_mix)
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
            n_samples = 10
        else:
            n_samples = len(lines)

        results = Parallel(n_jobs=-1, verbose=2)(
              delayed(main)(index, args.property) for index in range(n_samples))
        results = np.array(results)
        descriptors = results[:,0]
        descriptors = [descriptor for descriptor in descriptors]
        compounds_list = results[:,1]

        print('It took {} sec.'.format(time() - start))

        if args.descriptor_type == "mixtured":
            columns_list = ['Z_diff', 'Group_diff', 'kai(Pauling)_diff', \
                            'kai(Allen)_diff', 'EA_diff', 'IE1_diff', 'IE2_diff', \
                            'Rps-s_diff', 'Rvdw_diff', 'Rcov_diff', 'MP_diff', \
                            'BP_diff', 'Cp-g_diff','Cp-mol_diff', 'rho_diff', \
                            'E-fusion_diff', 'Thermal-Cond_diff', 'Mol-Vol_diff', \
                            'Z_ave', 'Group_ave', 'kai(Pauling)_ave', \
                            'kai(Allen)_ave', 'EA_ave', 'IE1_ave', 'IE2_ave', \
                            'Rps-s_ave', 'Rvdw_ave', 'Rcov_ave', 'MP_ave', \
                            'BP_ave', 'Cp-g_ave', 'Cp-mol_ave', 'rho_ave', \
                            'E-fusion_ave', 'Thermal-Cond_ave', 'Mol-Vol_ave']

        elif args.descriptor_type == "structure":
            columns_list = range(len(descriptors[0]))

        df_descriptors = pd.DataFrame(np.array(descriptors),
                                        columns=columns_list,
                                        index=compounds_list)
        df_descriptors.to_csv(args.save_path)
