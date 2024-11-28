#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import json
import metadata

def npz(path):
    with np.load(path.numpy()) as data:
        return data['prop'], data['remi'], data['c'], data['y']

def load_np(path, std, avg):
    p,r,c,y = tf.py_function(func=npz, inp=[path], Tout=(tf.float32, tf.float32, tf.float32, tf.float32))

    xs = tf.data.Dataset.zip(tf.data.Dataset.from_tensor_slices(p),
                                tf.data.Dataset.from_tensor_slices(r),
                                tf.data.Dataset.from_tensor_slices((c-avg)/std))
    ys = tf.data.Dataset.from_tensor_slices(y)

    return tf.data.Dataset.zip(xs,ys)

def get_dataset(directory):
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


    filename_dataset = tf.data.Dataset.list_files(str(directory/"*.npz"),shuffle=False).map(lambda x: (x, std, avg))
    return filename_dataset.flat_map(load_np)
