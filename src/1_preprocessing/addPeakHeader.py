import os
import pandas as pd
import numpy as np
import argparse
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument('--path_data', default="./data/",
                    help='Directory with peaks.tsv')



def main():
    args = parser.parse_args()
    path = args.path_data
    peak_path = os.path.join(path, "peaks.tsv")
    peaks = pd.read_csv(peak_path, sep="\t", header=None, index_col=None)
    peaks.columns = ["chrm", "start", "end", "stand"]
    peak_path = os.path.join(path, "peaks_with_header.tsv")
    peaks.to_csv(peak_path, sep="\t", index=None)

if __name__ == "__main__":
    main()
