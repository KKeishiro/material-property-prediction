import os
import sys
import numpy as np
import pandas as pd
import argparse
from time import time
from datetime import datetime

# Setting path------------------------------------------------------------------
dir_path = os.getcwd()
script_path = os.path.join(dir_path, 'script')
sys.path.append(script_path)
from atomic_rep_to_descriptors import *

descriptors_dir = dir_path + "/data/to_kanamori/cohesive/descriptors/"
compounds_list_cohesive = dir_path + "/data/to_kanamori/cohesive/compounds_name"
compounds_list_ltc = dir_path + "/data/to_kanamori/ltc/kappa"
compounds_list_mp = dir_path + "/data/to_kanamori/melting_temp/mp_data"
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="compute structure descriptors")
parser.add_argument("--property", required=True, help="property of a dataset")
parser.add_argument("--save_path", required=True, help="path to save data")
parser.add_argument("--isTest", action="store_true", help="whether to test or not")
args = parser.parse_args()


def main(index, property, path_lines):
    if property == "cohesive":
        compound_dir = path_lines[index].strip()
        descriptor_path = descriptors_dir + compound_dir + "/descriptors_st_bop"
    elif property == "ltc" or property == "mp":
        line = path_lines[index].strip().split()
        compound_dir = line[0]
        descriptor_path = dir_path + compound_dir + "/descriptors_st_bop"

    rep = []
    with open(descriptor_path) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            rep.append(lines[i].strip().split())

    # compute descriptors
    rep = np.array(rep).astype(np.float)
    descriptor = atomic_rep_to_compound_descriptor(rep, mom_order=2, icov=True)

    return descriptor, compound_dir


if __name__ == "__main__":

    if args.property == "cohesive":
        compounds_list_path = compounds_list_cohesive
    elif args.property == "ltc":
        compounds_list_path = compounds_list_ltc
    elif args.property == "mp":
        compounds_list_path = compounds_list_mp
    else:
        assert False, 'plased choose a valid property name'

    start = time()
    print('Started at {}'.format(datetime.now()))

    descriptors = []
    name_list = []

    with open(compounds_list_path) as f:
        lines = f.readlines()
        if args.isTest == True:
            n_samples = 10
        else:
            n_samples = len(lines)

        for i in range(n_samples):
            descriptor, compound_name = main(i, args.property, lines)
            descriptors.append(descriptor)
            name_list.append(compound_name)

        print('It took {} sec.'.format(time() - start))


        df_descriptors = pd.DataFrame(np.array(descriptors), index=name_list)
        df_descriptors.to_csv(args.save_path)
