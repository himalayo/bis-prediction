#!/usr/bin/env python3

import numpy as np
import os
import sys
from screen import bar, Status
import json

def case_metadata(path):
    with np.load(path) as data:
        age, sex, weight, height = data['c'][0]
        tot = data['c'].shape[0]
        return age*tot, sex*tot, weight*tot, height*tot, tot

def directory_metadata(current_directory=None, curr_status=None, avg_status=None, status_bar=None):

    if status_bar is None:
        status_bar = Status("Progress")

    if current_directory is None:
        current_directory = Status("Current directory")

    if curr_status is None:
        curr_status = Status("Current case")

    if avg_status is None:
        avg_status = Status("Avg")


    def process_directory(directory):
        current_directory.update(directory)
        sum_age = 0
        sum_sex = 0
        sum_wt  = 0
        sum_ht  = 0


        sum_age_sq = 0
        sum_sex_sq = 0
        sum_wt_sq  = 0
        sum_ht_sq  = 0

        tot     = 0
        cum_tot = {}
        paths = [(os.path.join(directory,path),) for path in sorted(os.listdir(directory))]
        for i,path,(case_age, case_sex, case_wt, case_ht, case_samples) in status_bar.bar()(case_metadata)(paths):
            curr_status.update([case_age, case_sex, case_wt, case_samples])
            sum_age += case_age
            sum_sex += case_sex
            sum_wt  += case_wt
            sum_ht  += case_ht

            sum_age_sq += ((case_age/case_samples)**2)*case_samples
            sum_sex_sq += ((case_sex/case_samples)**2)*case_samples
            sum_wt_sq  += ((case_wt/case_samples)**2)*case_samples
            sum_ht_sq  += ((case_ht/case_samples)**2)*case_samples

            cum_tot[tot] = path[0]
            tot += case_samples
            avg_status.update([sum_age/tot, sum_sex/tot, sum_wt/tot, sum_ht/tot, tot])
        return {
            'avg_age': sum_age/tot,
            'avg_sex': sum_sex/tot,
            'avg_wt' : sum_wt/tot,
            'avg_ht' : sum_ht/tot,
            'samples': tot,
            'sum_age_sq': sum_age_sq,
            'sum_sex_sq': sum_sex_sq,
            'sum_wt_sq' : sum_wt_sq,
            'sum_ht_sq' : sum_ht_sq,
            'sum_age': sum_age,
            'sum_sex': sum_sex,
            'sum_wt' : sum_wt,
            'sum_ht' : sum_ht,
            'range' : cum_tot}

    return process_directory

def dataset_metadata(directory):
    status_bar       = Status("Progress")
    current_directory= Status("Current directory")
    curr_status      = Status("Case metadata")
    avg_status       = Status("Avg")

    process_directory = directory_metadata(current_directory = current_directory,
                                           curr_status = curr_status,
                                           avg_status = avg_status,
                                           status_bar = status_bar)

    xs = {}
    for subdir in os.listdir(directory):
        xs[subdir] = process_directory(os.path.join(directory,subdir))

    xs['total'] = {
        'avg_age': (xs['train']['sum_age'] + xs['val']['sum_age'])
                   / (xs['train']['samples']+xs['val']['samples']),
        'avg_sex': (xs['train']['sum_sex'] + xs['val']['sum_sex'])
                   / (xs['train']['samples']+xs['val']['samples']),

        'avg_wt' : (xs['train']['sum_wt'] + xs['val']['sum_wt'])
                   / (xs['train']['samples']+xs['val']['samples']),
        'avg_ht' : (xs['train']['sum_ht'] + xs['val']['sum_ht'])
                   / (xs['train']['samples']+xs['val']['samples']),

        'std_age': ((xs['train']['sum_age_sq']+xs['val']['sum_age_sq']-
                    (((xs['train']['sum_age']+xs['val']['sum_age'])**2)/
                     (xs['train']['samples']+xs['val']['samples'])))/
                    (xs['train']['samples']+xs['val']['samples']-1))**(1/2),

        'std_sex': ((xs['train']['sum_sex_sq']+xs['val']['sum_sex_sq']-
                    (((xs['train']['sum_sex']+xs['val']['sum_sex'])**2)/
                     (xs['train']['samples']+xs['val']['samples'])))/
                    (xs['train']['samples']+xs['val']['samples']-1))**(1/2),

        'std_wt' : ((xs['train']['sum_wt_sq']+xs['val']['sum_wt_sq']-
                    (((xs['train']['sum_wt']+xs['val']['sum_wt'])**2)/
                     (xs['train']['samples']+xs['val']['samples'])))/
                    (xs['train']['samples']+xs['val']['samples']-1))**(1/2),

        'std_ht' : ((xs['train']['sum_ht_sq']+xs['val']['sum_ht_sq']-
                    (((xs['train']['sum_ht']+xs['val']['sum_ht'])**2)/
                     (xs['train']['samples']+xs['val']['samples'])))/
                    (xs['train']['samples']+xs['val']['samples']-1))**(1/2)
    }

    return xs



if __name__ == "__main__":
    print(case_metadata("data/train/001.npz"))
    metadata = dataset_metadata("data")
    with open("metadata.json", "w+") as f:
        json.dump(metadata, f)
    print("\n",json.dumps(metadata))
