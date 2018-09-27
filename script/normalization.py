import numpy as np
import pandas as pd
import argparse
from data_processing import normalization

# Setting path------------------------------------------------------------------
# dir_path = "/Users/keishiro/Documents/M2_research" # lab's laptop
dir_path = "/Users/user/Documents/M2_research" # my macbook
# ------------------------------------------------------------------------------

# load encoded DataFrame
'''
In order to run this program, encoded data needs to be obtained first.
'''
encoded_df_cohesive = pd.read_csv(dir_path + "/data/seko/coh_energy/encoded_compounds.csv", index_col=0)
encoded_df_ltc = pd.read_csv(dir_path + "/data/seko/ltc/encoded_compounds.csv", index_col=0)
encoded_df_mp = pd.read_csv(dir_path + "/data/seko/mp/encoded_compounds.csv", index_col=0)

parser = argparse.ArgumentParser(description="normalize descriptors")
parser.add_argument("--property", required=True, help="property of a dataset")
parser.add_argument("--file_path", required=True,
                    help="path to a target file. Make sure that property of the \
                    target file is the same with --property param.")
parser.add_argument("--save_path", required=True, help="path to save data")
args = parser.parse_args()


if __name__ == "__main__":
    if args.property == "cohesive":
        encoded_df = encoded_df_cohesive
    elif args.property == "ltc":
        encoded_df = encoded_df_ltc
    elif args.property == "mp":
        encoded_df = encoded_df_mp
    target_df = pd.read_csv(args.file_path, index_col=0)
    normalization(target_df, encoded_df, args.save_path)
