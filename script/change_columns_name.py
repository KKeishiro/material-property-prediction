#!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="change columns name")
parser.add_argument("--suffix", required=True, help="what is put at the end of column name")
parser.add_argument("--file_path", required=True, help="path to a target file")
parser.add_argument("--save_path", required=True, help="path to save data")
args = parser.parse_args()

def change_name(df, suffix):
    for column in df.columns:
        df.rename(columns={column: column + suffix}, inplace=True)
    return df

def init_name(df_review, df):
    assert len(df_review.values) == len(df.values)
    for i in range(len(df.columns)):
        df.rename(columns={df.columns[i]: df_review.columns[i]}, inplace=True)
    return df

if __name__ == "__main__":
    df = change_name(args.file_path, args.suffix)
    df.to_csv(args.save_path)
