#!/usr/bin/env python3
import os
import re
import csv
import pickle
from numba import njit, prange
import numpy as np
import pandas as pd
import statsmodels.api as sm
import keras
import lowess_cuda

def load_data_from(path):
    data = pd.read_csv(path).convert_dtypes().astype({'Case ID': 'category'})
    data = data.rename(columns=dict(zip(data.columns,[re.sub(r"\(.*?\)","",string).split(' ')[0].lower() for string in data.columns])))
    data = data.rename(columns={'case':'id'})

    return data.groupby('id')

def load_data(path, url=None):
    if os.path.isfile(path):
        print(f"{path} found, loading from local file...")
        return load_data_from(path)
    if not url is None:
        print(f"{path} was not found, loading from {url}")
        return load_data_from(url)
    raise Exception("File does not exist, and a download url was not provided.")


def smooth_case(vals):
    vals_padded = np.pad(vals.to_numpy(), (100, 100), 'edge')
    return sm.nonparametric.lowess(vals_padded, np.arange(-100, vals.shape[0] + 100), frac=0.03)[100:-100, 1]

@njit(parallel=True, fastmath=True)
def process_case(remifentanil, propofol, sqi, start, end):
    n    = propofol.shape[0]
    vals = np.zeros((2,n, start.shape[0]))

    for i in prange(n):
        if sqi[i] <= 50:
            continue

        istart = np.maximum(i + 1 - start, np.zeros_like(start))
        iend   = np.maximum(i + 1 - end, np.zeros_like(end))
        for j in prange(start.shape[0]):
            vals[0][i][j] = np.sum(propofol[istart[j]:iend[j]])
            vals[1][i][j] = np.sum(remifentanil[istart[j]:iend[j]])
    return vals

def make_sequence(raw_data, timepoints):
    gaps = np.arange(0, 10 * (timepoints + 1), 10)
    start_sequence, end_sequence = np.array(list(reversed(list(zip(gaps[1:], gaps))))).T
    os.makedirs(os.path.join("data","train"), exist_ok=True)
    os.makedirs(os.path.join("data","test"), exist_ok=True)
    os.makedirs(os.path.join("data","val"), exist_ok=True)
    for id, case in raw_data:  # for each case
        print(f"beginning case {id}...")
        ifirst = -1
        ilast = -1
        ppf_seq = []
        rft_seq = []
        if id[:5] == "train":
            print(f"Calculating smoothed BIS for case {id}...")
            bis =  np.clip(smooth_case(case['bis']) / 100, 0, 1)
            print(f"Calculating smoothed BIS for case {id}: Done")
        else:
            bis = case['bis'] / 100

        sqi = case['sqi'].to_numpy()
        ppf_dose = case['propofol'].to_numpy()/12.0
        rft_dose = case['remifentanil'].to_numpy()/12.0
        age = case['age'].to_numpy()
        sex = case['sex'].to_numpy()
        wt =  case['weight'].to_numpy()
        ht =  case['height'].to_numpy()
        c  = np.stack((age, sex, wt, ht), -1)
        print(f"processing infusions for case {id}...")
        prop, remi = process_case(rft_dose, ppf_dose, sqi, start_sequence, end_sequence)

        print(f"saving case {id}...")
        np.savez(os.path.join("data",id.replace('_','/')), prop=prop, remi=remi, y=bis, c=c)
        print(f"case {id}: Done")

if __name__ == "__main__":
    print("Loading training data...")
    data = load_data("data.csv", url="https://osf.io/download/y5kcx/")
    print("Loading training data: Done")
    print("Storing all cases...")
    make_sequence(data, 180)
    print("Storing all cases: Done")
