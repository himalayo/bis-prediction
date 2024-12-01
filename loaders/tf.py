#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import json
import metadata
import dataset_processing
import pathlib

def npz(path):
    with np.load(path.numpy()) as data:
        return data['prop'].reshape([data['prop'].shape[0], data['prop'].shape[1],1]), data['remi'].reshape([data['remi'].shape[0], data['remi'].shape[1],1]),data['c'],data['y']

def get_dataset(directory):
    try:
        try:
            with open(directory.parent / "metadata.json", "r") as f:
                meta = json.load(f)
        except:
            meta = metadata.dataset_metadata(directory.parent)
            with open(directory.parent / "metadata.json", "w+") as f:
                json.dump(meta, f)

        avg = tf.constant((meta['total']['avg_age'], meta['total']['avg_sex'],
                    meta['total']['avg_wt'], meta['total']['avg_ht']))
        std = tf.constant((meta['total']['std_age'], meta['total']['std_sex'],
                        meta['total']['std_wt'], meta['total']['std_ht']))


        filename_dataset = tf.data.Dataset.list_files(str(directory/"*.npz"),shuffle=False)
        datasets = []
        for path in filename_dataset:
            p,r,c,y = npz(path)
            xs = tf.data.Dataset.zip(
                tf.data.Dataset.from_tensor_slices(p),
                tf.data.Dataset.from_tensor_slices(r),
                tf.data.Dataset.from_tensor_slices((c-avg)/std)
            )

            ys = tf.data.Dataset.from_tensor_slices(y)

            datasets.append(tf.data.Dataset.zip(xs, ys))

        return tf.data.Dataset.sample_from_datasets(datasets).batch(64)
    except Exception as e:
        print(e)
        data = dataset_processing.load_data("data.csv", url="https://osf.io/download/y5kcx")
        print("\n"*10)
        dataset_processing.make_sequence(data)

        if directory.exists():
            return get_dataset(directory)
        else:
            raise Exception(f"{directory} is not a directory")
