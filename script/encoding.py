import os
import numpy as np
import pandas as pd
import argparse
from getdata_from_poscar import DataConstruction
from data_processing import composition_encoding

# Setting path------------------------------------------------------------------
dir_path = os.getcwd()

descriptors_dir = dir_path + "/data/to_kanamori/cohesive/descriptors/"
compounds_list_cohesive = dir_path + "/data/to_kanamori/cohesive/compounds_name"
compounds_list_ltc = dir_path + "/data/to_kanamori/ltc/kappa"
compounds_list_mp = dir_path + "/data/to_kanamori/melting_temp/mp_data"
# ------------------------------------------------------------------------------
atomic_df = pd.read_csv(dir_path + "/data/atomic_data_reduced.csv", index_col=0)
atomic_df = atomic_df.drop(["Rps-d"], axis=1)

parser = argparse.ArgumentParser(description="encode compounds w.r.t. composition")
parser.add_argument("--property", required=True, help="property of a dataset")
parser.add_argument("--save_path", required=True, help="path to save encoded data")
parser.add_argument("--isTest", action="store_true", help="whether to test or not")
args = parser.parse_args()


if __name__ == "__main__":
    #set empty DataFrame for encoding
    encoded = pd.DataFrame(index=[], columns=atomic_df.index)

    if args.property == "cohesive":
        compounds_list_path = compounds_list_cohesive
    elif args.property == "ltc":
        compounds_list_path = compounds_list_ltc
    elif args.property == "mp":
        compounds_list_path = compounds_list_mp
    else:
        assert False, 'plased choose a valid property name'

    with open(compounds_list_path) as f:
        lines = f.readlines()
        if args.isTest == True:
            n_samples = 10
        else:
            n_samples = len(lines)

        for i in range(n_samples):
            if i % 500 == 0:
                print(str(i) + "compounds are done!")

            if args.property == "cohesive":
                compound_dir = lines[i].strip()
                POSCAR_path = descriptors_dir + compound_dir + "/POSCAR"
            elif args.property == "ltc" or args.property == "mp":
                line = lines[i].strip().split()
                compound_dir = line[0]
                POSCAR_path = dir_path + compound_dir + "/POSCAR"

            # make encoded DataFrame
            dc = DataConstruction()
            structure = dc.set_structure_from_poscar_file(POSCAR_path)
            primitive_structure = structure.get_primitive_structure()
            species = primitive_structure.species
            df = composition_encoding(atomic_df, species, compound_dir)
            encoded = encoded.append(df)
        encoded.to_csv(args.save_path)
