#!/usr/bin/env python3

import sys
import pathlib
import keras
import numpy as np
import loaders.keras
import models.lstm
import json
import matplotlib.pyplot as plt

def evaluate(model):
    with open("data/metadata.json") as meta:
        metadata = json.load(meta)
        ranges = metadata['test']['range']

    avg = np.array((metadata['total']['avg_age'],metadata['total']['avg_sex'],
                    metadata['total']['avg_wt'], metadata['total']['avg_ht']))
    std = np.array((metadata['total']['std_age'],metadata['total']['std_sex'],
                    metadata['total']['std_wt'], metadata['total']['std_ht']))


    for i,(x,y) in enumerate(zip(ranges.keys(),list(ranges.keys())[1:])):
        tensorfile = np.load(ranges[x])
        data = dict(map(lambda key: (key, np.array(tensorfile[key])), tensorfile))
        data['c'] -= avg
        data['c'] /= std

        predicted_bis = np.squeeze(model((data['prop'], data['remi'], data['c'])).numpy())
        plt.figure()
        plt.plot(predicted_bis)
        plt.plot(data['y'])
        plt.show()

        print(i, predicted_bis, data['y'])
        with open(f'output/{i}.json', 'w+') as f:
            json.dump({'predicted': predicted_bis.tolist(), 'expected': data['y'].tolist()}, f)

def train(model):
    if model == "lstm":
        train_ds = loaders.keras.BISDataset(pathlib.Path("data/train"))
        val_ds   = loaders.keras.BISDataset(pathlib.Path("data/val"))

        model = models.lstm.create_model()
        hist, model = models.lstm.train(model, train_dataset=train_ds, val_dataset=val_ds)
        model.save("bis_lstm.keras")

if __name__ == "__main__":

    for command,argument in zip(sys.argv[1:], sys.argv[2:]):
        if command == "--train" or command == "-t":
            train(argument)
        elif command == "--evaluate" or command == "-e":
            evaluate(keras.saving.load_model(argument))
