#!/usr/bin/env python3

import torch
import numpy as np
import json
import metadata
import os
import pathlib

class BISDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        try:
            with open(directory / "metadata.json", "r") as f:
                self.metadata = json.load(f)
        except:
            self.metadata = metadata.dataset_metadata(os.path.abspath(os.path.join(directory, os.pardir)))
            with open(directory / "metadata.json", "w+") as f:
                json.dump(self.metadata, f)

        self.tensors = {}
        self.directory = pathlib.Path(directory)
        self.z = torch.tensor((self.metadata['total']['avg_age'], self.metadata['total']['avg_sex'],
                           self.metadata['total']['avg_wt'], self.metadata['total']['avg_ht']))
        self.size = self.metadata[self.directory.name]['samples']
        self.ranges = dict(map(lambda x: (int(x[0]), x[1]), self.metadata[self.directory.name]['range'].items()))


    def __len__(self):
        return self.metadata[self.directory.name]['samples']

    def __getitem__(self, idx):
        for x,y in zip(self.ranges.keys(),list(self.ranges.keys())[1:]):
            if x <= idx < y:
                if x in self.tensors.keys():
                    return (self.tensors[x]['prop'][idx-x], self.tensors[x]['remi'][idx-x], self.tensors[x]['c'][idx-x]), self.tensors[x]['y'][idx-x]
                tensorfile = np.load(self.ranges[x])
                self.tensors[x] = dict(map(lambda key: (key, torch.tensor(tensorfile[key])), tensorfile))
                return (self.tensors[x]['prop'][idx-x], self.tensors[x]['remi'][idx-x], self.tensors[x]['c'][idx-x]), self.tensors[x]['y'][idx-x]

        x = list(self.ranges.keys())[-1]
        if x in self.tensors.keys():
            return (self.tensors[x]['prop'][idx-x], self.tensors[x]['remi'][idx-x], self.tensors[x]['c'][idx-x]), self.tensors[x]['y'][idx-x]

        tensorfile = np.load(self.ranges[x])
        self.tensors[x] = dict(map(lambda key: (key, torch.tensor(tensorfile[key])), tensorfile))
        return (self.tensors[x]['prop'][idx-x], self.tensors[x]['remi'][idx-x], self.tensors[x]['c'][idx-x]), self.tensors[x]['y'][idx-x]
