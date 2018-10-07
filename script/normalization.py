import os
import numpy as np
import pandas as pd
import argparse
from data_processing import normalization

# Setting path------------------------------------------------------------------
dir_path = os.getcwd()
# ------------------------------------------------------------------------------

# load encoded DataFrame
'''
In order to run this program, encoded data needs to be obtained first.
'''
encoded_df_cohesive = pd.read_csv(dir_path + "/data/descriptors/cohesive/encoded_compounds.csv", index_col=0)
encoded_df_ltc = pd.read_csv(dir_path + "/data/descriptors/ltc/encoded_compounds.csv", index_col=0)
encoded_df_mp = pd.read_csv(dir_path + "/data/descriptors/mp/encoded_compounds.csv", index_col=0)

parser = argparse.ArgumentParser(description="normalize descriptors")
parser.add_argument("--property", required=True, help="property of a dataset")
parser.add_argument("--file_path", required=True,
                    help="path to a target file. Make sure that property of the \
                    target file is the same with --property param.")
args = parser.parse_args()


if __name__ == "__main__":
    if args.property == "cohesive":
        encoded_df = encoded_df_cohesive
    elif args.property == "ltc":
        encoded_df = encoded_df_ltc
    elif args.property == "mp":
        encoded_df = encoded_df_mp
    else:
        assert False, 'please choose a valid property name'

    df = pd.read_csv(args.file_path, index_col=0)
    save_path = os.path.splitext(args.file_path)[0] + '_norm.csv'
    normalization(df, encoded_df, save_path)
