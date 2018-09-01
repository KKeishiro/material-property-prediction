import numpy as np
import pandas as pd
import argparse
from data_processing import compute_ave_descriptors, compute_std_descriptors

# Setting path------------------------------------------------------------------
# dir_path = "/Users/keishiro/Documents/M2_research" # lab's laptop
dir_path = "/Users/user/Documents/M2_research" # my macbook
# ------------------------------------------------------------------------------

# load atomic DataFrame
atomic_df = pd.read_csv(dir_path + "/data/seko/atomic_data_20160603.csv", index_col=0)
atomic_df = atomic_df.drop(["Rps-d"], axis=1)

# load encoded DataFrame
'''
In order to get atomic descriptors, encoded data needs to be obtained first.
'''
encoded_df_cohesive = pd.read_csv(dir_path + "/data/seko/coh_energy/encoded_compounds.csv", index_col=0)
encoded_df_ltc = pd.read_csv(dir_path + "/data/seko/ltc/encoded_compounds.csv", index_col=0)
encoded_df_mp = pd.read_csv(dir_path + "/data/seko/mp/encoded_compounds.csv", index_col=0)

# load atomic average DataFrame
'''
In order to get std descriptors, average descriptors needs to be computed first.
When you compute average descriptors, comment out the following lines and
line 43, 46 and 49.
'''
average_df_cohesive = pd.read_csv(dir_path + "/data/seko/coh_energy/X_element_ave.csv", index_col=0)
average_df_ltc = pd.read_csv(dir_path + "/data/seko/ltc/X_element_ave.csv", index_col=0)
average_df_mp = pd.read_csv(dir_path + "/data/seko/mp/X_element_ave.csv", index_col=0)

parser = argparse.ArgumentParser(description="compute atomic descriptors")
parser.add_argument("--property", required=True, help="property of a dataset")
parser.add_argument("--descriptor_type", default="mean", help="descriptors type")
parser.add_argument("--save_path", required=True, help="path to save data")
args = parser.parse_args()


if __name__ == "__main__":
    if args.property == "cohesive":
        encoded_df = encoded_df_cohesive
        average_df = average_df_cohesive
    elif args.property == "ltc":
        encoded_df = encoded_df_ltc
        average_df = average_df_ltc
    elif args.property == "mp":
        encoded_df = encoded_df_mp
        average_df = average_df_mp

    if args.descriptor_type == "mean":
        compute_ave_descriptors(atomic_df, encoded_df, args.save_path)
    elif args.descriptor_type == "std":
        compute_std_descriptor(atomic_df, average_df, encoded_df, args.save_path)
