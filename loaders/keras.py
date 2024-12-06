#!/usr/bin/env python3
import keras
import metadata
import numpy as np
import json
import pathlib
import math

class BISDataset(keras.utils.PyDataset):
    def __init__(self, directory, batch_size=64, **kwargs):
        super().__init__(**kwargs)
        try:
            with open(directory.parent / "metadata.json", "r") as f:
                self.metadata = json.load(f)
        except:
            print(directory.parent)
            self.metadata = metadata.dataset_metadata(directory.parent)
            with open(directory.parent / "metadata.json", "w+") as f:
                json.dump(self.metadata, f)

        self.tensors = {}
        self.directory = pathlib.Path(directory)
        self.avg = np.array((self.metadata['total']['avg_age'], self.metadata['total']['avg_sex'],
                           self.metadata['total']['avg_wt'], self.metadata['total']['avg_ht']))
        self.std = np.array((self.metadata['total']['std_age'], self.metadata['total']['std_sex'],
                           self.metadata['total']['std_wt'], self.metadata['total']['std_ht']))
        self.size = self.metadata[self.directory.name]['samples']
        self.ranges = dict(map(lambda x: (int(x[0]), x[1]), self.metadata[self.directory.name]['range'].items()))
        self.batch_size = batch_size



    def __len__(self):
        return math.ceil(self.metadata[self.directory.name]['samples']/self.batch_size)


    def single_example(self, idx):
        out = lambda x: ((self.tensors[x]['prop'][idx-x].reshape(1,180,1), self.tensors[x]['remi'][idx-x].reshape(1,180,1), self.tensors[x]['c'][idx-x].reshape(1,4)), self.tensors[x]['y'][idx-x].reshape((1,)))

        for x,y in zip(self.ranges.keys(),list(self.ranges.keys())[1:]):
            if x <= idx < y:
                if x in self.tensors.keys():
                    return out(x)
                tensorfile = np.load(self.ranges[x])
                self.tensors[x] = dict(map(lambda key: (key, np.array(tensorfile[key])), tensorfile))
                self.tensors[x]['c'] -= self.avg
                self.tensors[x]['c'] /= self.std
                return out(x)

        x = list(self.ranges.keys())[-1]
        if x in self.tensors.keys():
            return out(x)

        tensorfile = np.load(self.ranges[x])
        self.tensors[x] = dict(map(lambda key: (key, np.array(tensorfile[key])), tensorfile))
        self.tensors[x]['c'] -= self.avg
        self.tensors[x]['c'] /= self.std
        return out(x)

    def __getitem__(self, idx):
        out_p, out_r, out_c, out_y = [],[],[],[]
        for i in range(self.batch_size):
            try:
                (p,r,c),y = self.single_example((idx*self.batch_size)+i)
                out_p.append(p)
                out_r.append(r)
                out_c.append(c)
                out_y.append(y)
            except:
                pass
        return (np.concatenate(out_p, 0), np.concatenate(out_r, 0), np.concatenate(out_c, 0)), np.concatenate(out_y, 0)
