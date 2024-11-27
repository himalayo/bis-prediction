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
from screen import log_action, bar, Status

gaps = np.arange(0, 10 * 181, 10)
start_sequence, end_sequence = np.array(list(reversed(list(zip(gaps[1:], gaps))))).T

def load_data_from(path):
    data = pd.read_csv(path).convert_dtypes().astype({'Case ID': 'category'})
    data = data.rename(columns=dict(zip(data.columns,[re.sub(r"\(.*?\)","",string).split(' ')[0].lower() for string in data.columns])))
    data = data.rename(columns={'case':'id'})

    return data.groupby('id')

def load_data_from_file(path):
    return load_data_from(path)

def load_data_from_url(url):
    return load_data_from(url)

@log_action("Loading data")
def load_data(path, url=None):
    if os.path.isfile(path):
        return load_data_from_file(path)

    if not url is None:
        return load_data_from_url(url)
    raise Exception("File does not exist, and a download url was not provided.")


def smooth_case(id, vals):
    vals_padded = np.pad(vals.to_numpy(), (100, 100), 'edge')
    return np.clip(sm.nonparametric.lowess(vals_padded,
                                           np.arange(-100, vals.shape[0] + 100),
                                           frac=0.03)[100:-100, 1]/100,
                   0,
                   1)

@njit(parallel=True, fastmath=True)
def process_infusions(id, remifentanil, propofol, sqi, start, end):
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

def save_data(id, prop, remi, bis, c):
    np.savez(os.path.join("data",id.replace('_','/')), prop=prop, remi=remi, y=bis, c=c)

def process_case(status):

    def process(id, case):
        status.update("Starting")
        ifirst = -1
        ilast = -1
        ppf_seq = []
        rft_seq = []
        if id[:5] == "train":
            status.update("Calculating smoothed bis")
            bis = smooth_case(id, case['bis'])
        else:
            bis = case['bis'] / 100

        status.update("Converting to numpy")
        sqi = case['sqi'].to_numpy()
        ppf_dose = case['propofol'].to_numpy()/12.0
        rft_dose = case['remifentanil'].to_numpy()/12.0
        age = case['age'].to_numpy()
        sex = case['sex'].to_numpy()
        wt =  case['weight'].to_numpy()
        ht =  case['height'].to_numpy()
        c  = np.stack((age, sex, wt, ht), -1)

        status.update("Processing infusions")
        prop, remi = process_infusions(id, rft_dose, ppf_dose, sqi, start_sequence, end_sequence)
        status.update("Done")
        return prop, remi, bis, c

    return process


def make_sequence(raw_data):
    avg_age = 0
    avg_sex = 0
    avg_wt = 0
    avg_ht = 0
    tot    = 0


    os.makedirs(os.path.join("data","train"), exist_ok=True)
    os.makedirs(os.path.join("data","test"), exist_ok=True)
    os.makedirs(os.path.join("data","val"), exist_ok=True)
    progress = Status(title="Cases")
    current_case = Status(title="Current case")
    case_status = Status(title="Case status")
    case_avg = Status(title="Avg")
    case_age = Status(title="Age")
    case_sex = Status(title="Sex")
    case_wt = Status(title="Weight")
    case_ht = Status(title="Height")
    case_tot = Status(title="Samples")
    print("\033[2J")

    for _,(id, case), (prop, remi, bis, c) in progress.bar(bar_size=50)(process_case(case_status))(raw_data):
        current_case.update(id)
        case_age.update(c[0,0])
        case_sex.update(c[0,1])
        case_wt.update(c[0,2])
        case_ht.update(c[0,3])
        case_tot.update(c.shape[0])
        case_status.update("Saving")
        save_data(id, prop, remi, bis, c)
        avg_age += case_age.data*case_tot.data
        avg_sex += case_sex.data*case_tot.data
        avg_wt  += case_wt.data* case_tot.data
        avg_ht  += case_ht.data* case_tot.data
        tot     += case_tot.data

        case_avg.update([avg_age/tot, avg_sex/tot, avg_wt/tot, avg_ht/tot, tot])

if __name__ == "__main__":
    data = load_data("data.csv", url="https://osf.io/download/y5kcx")
    print("\n"*10)
    make_sequence(data)
