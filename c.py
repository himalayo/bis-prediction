"""
Predicting bispectral index from propofol, remifentanil infusion history.

This program demonstrates how to build an estimation model of drug effect using deep learning techniques.
It runs on python 3.5.2 with keras 1.2.2 and tensorflow 1.2.1.

Developed by Hyung-Chul Lee (lucid80@gmail.com) in Aug 2016 and
licensed to the public domain for academic advancement.
"""
import os
import csv
import pickle
import numpy as np
import statsmodels.api as sm
import keras
#from keras import layers
#from keras import callbacks
#from keras import Model, Sequential, load_model
#from keras.layers import Dense, Dropout, LSTM, Input, merge
#from keras.callbacks import ModelCheckpoint, EarlyStopping

# parameters
timepoints = 180
lnode = 8
fnode = 16
batch_size = 64
cache_path = "cache.var"
output_dir = "output"
weight_path = output_dir + "/weights.hdf5"

# create output dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# load data and generate sequence data
if os.path.exists(cache_path):
    train_p, train_r, train_c, train_y, val_p, val_r, val_c, val_y, test_p, test_r, test_c, test_y = pickle.load(open(cache_path, "rb"))
else:
    train_p = []
    train_r = []
    train_c = []
    train_y = []
    val_p = []
    val_r = []
    val_c = []
    val_y = []
    test_p = {} # by case
    test_r = {}
    test_c = {}
    test_y = {}

    # load raw data
    f = open('data.csv', 'rt')
    raw_data = {}
    first = True
    for row in csv.reader(f):
        id = row[0]
        if first:
            first = False
            continue
        if id not in raw_data:
            raw_data[id] = []
        rowvals = []
        for j in range(1, len(row)):
            rowvals.append(float(row[j]))
        raw_data[id].append(rowvals)

    # smooth bis
    smoothed = {}
    for id, case in raw_data.items():
        case_len = len(case)
        vals = [row[1] for row in case]  # bis values
        vals_padded = [vals[0]] * 100 + vals + [vals[-1]] * 100
        smoothed[id] = sm.nonparametric.lowess(vals_padded, np.arange(-100, case_len + 100), frac=0.03)[100:-100, 1]

    # make sequence
    gaps = np.arange(0, 10 * (timepoints + 1), 10)
    for id, case in raw_data.items():  # for each case
        ifirst = -1
        ilast = -1
        ppf_seq = []
        rft_seq = []

        is_test = id[:4] == "test"
        if is_test:
            case_p= []
            case_r = []
            case_c = []
            case_y = []

        for isamp in range(len(case)):
            row = case[isamp]

            bis = row[1] / 100  # normalize bis
            bis_smooth = max(0, min(1, smoothed[id][isamp] / 100))
            sqi = row[2]
            ppf_dose = row[3]
            rft_dose = row[4]
            age = row[5]
            sex = row[6]
            wt = row[7]
            ht = row[8]
            ppf_seq.append(ppf_dose)  # make time sequence
            rft_seq.append(rft_dose)

            if ifirst is None:  # before started
                if ppf_dose < 1:
                    continue
                ifirst = isamp
            else:  # started
                if ppf_dose > 0.05:  # restarted
                    ilast = isamp
            if ilast is not None and isamp - ilast > gaps[-1]:
                break  # case finished

            if sqi <= 50:
                continue

            pvals = []
            rvals = []
            for i in reversed(range(timepoints)):
                istart = isamp + 1 - gaps[i + 1]
                iend = isamp + 1 - gaps[i]
                pvals.append(sum(ppf_seq[max(0, istart):max(0, iend)]))
                rvals.append(sum(rft_seq[max(0, istart):max(0, iend)]))

            if id[:5] == "train":
                train_p.append(pvals)
                train_r.append(rvals)
                train_c.append([age, sex, wt, ht])
                train_y.append(bis_smooth)  # smoothed bis for training set
            elif id[:3] == "val":
                val_p.append(pvals)
                val_r.append(rvals)
                val_c.append([age, sex, wt, ht])
                val_y.append(bis)
            elif id[:4] == "test":
                case_p.append(pvals)
                case_r.append(rvals)
                case_c.append([age, sex, wt, ht])
                case_y.append(bis)

        if is_test:
            test_p[id] = case_p
            test_r[id] = case_r
            test_c[id] = case_c
            test_y[id] = case_y

    # save cache file
    pickle.dump((train_p, train_r, train_c, train_y, val_p, val_r, val_c, val_y, test_p, test_r, test_c, test_y), open(cache_path, "wb"), protocol=4)

train_val_c = train_c + val_c

# convert data to numpy array
train_p = np.array(train_p)
train_r = np.array(train_r)
train_c = np.array(train_c)
train_y = np.array(train_y)
train_p = train_p.reshape(train_p.shape[0], train_p.shape[1], 1)
train_r = train_r.reshape(train_r.shape[0], train_r.shape[1], 1)

val_p = np.array(val_p)
val_r = np.array(val_r)
val_c = np.array(val_c)
val_y = np.array(val_y)
val_p = val_p.reshape(val_p.shape[0], val_p.shape[1], 1)
val_r = val_r.reshape(val_r.shape[0], val_r.shape[1], 1)

# normalize data
mean_c = np.mean(train_val_c, axis=0)
std_c = np.std(train_val_c, axis=0)
train_c = (train_c - mean_c) / std_c
train_p /= 12.0
train_r /= 12.0
val_c = (val_c - mean_c) / std_c
val_p /= 12.0
val_r /= 12.0

train_x = [train_p, train_r, train_c]
val_x = [val_p, val_r, val_c]

if os.path.exists(weight_path):
    model = keras.load_model(weight_path)
else:
    # build a model
    cin = keras.layers.Input(batch_shape=(None, 4))
    pin = keras.layers.Input(batch_shape=(None, timepoints, 1))
    rin = keras.layers.Input(batch_shape=(None, timepoints, 1))
    cout = cin
    pout = pin
    rout = rin
    pout = keras.layers.LSTM(lnode, input_shape=(timepoints, 1), activation='tanh')(pout)
    rout = keras.layers.LSTM(lnode, input_shape=(timepoints, 1), activation='tanh')(rout)
    out = keras.layers.merge([pout, rout, cout], mode='concat')
    out = keras.layers.Dense(fnode, init='he_uniform', activation='relu', kernel_regularizer=keras.regularizers.L2(1.0e-4))(out)
    out = keras.layers.Dropout(0.5)(out)
    out = keras.layers.Dense(1, init='he_uniform', activation='sigmoid')(out)

    model = keras.Model(input=[pin, rin, cin], output=[out])
    model.compile(loss='mae', optimizer='adam')

    # train the model
    hist = model.fit(train_x, train_y, validation_data=(val_x, val_y), nb_epoch=10, batch_size=batch_size, shuffle=True,
                     callbacks=[keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=weight_path, verbose=1, save_best_only=True),
                                keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='auto')])
    model.load_weights(weight_path)

# test model
sum_err = 0
cnt_err = 0
print('id\ttesting_err')
for id in test_p.keys():
    case_p = np.array(test_p[id])
    case_r = np.array(test_r[id])
    case_c = np.array(test_c[id])
    true_y = np.array(test_y[id])
    true_y = np.array(true_y)

    case_p = case_p.reshape(case_p.shape[0], case_p.shape[1], 1)
    case_r = case_r.reshape(case_r.shape[0], case_r.shape[1], 1)
    case_c = (case_c - mean_c) / std_c
    case_p /= 12.0
    case_r /= 12.0

    case_x = [case_p, case_r, case_c]
    pred_y = model.predict(case_x)

    true_y = true_y.T
    pred_y = pred_y[:, 0].T

    err = np.mean(np.abs(np.subtract(true_y, pred_y)))

    true_y = true_y.tolist()
    pred_y = pred_y.tolist()

    case_len = len(pred_y)
    sum_err += err * case_len
    cnt_err += case_len

    print('{}\t{}'.format(id, err))

    # save the result
    fo = open('{}/{}.csv'.format(output_dir, id), 'wt')
    writer = csv.writer(fo, dialect='excel', lineterminator='\n')
    header = ['Measured BIS', 'Expected BIS']
    writer.writerow(header)
    for i in range(len(true_y)):
        writer.writerow([true_y[i], pred_y[i]])

if cnt_err > 0:
    print("Mean test error: {}".format(sum_err / cnt_err))
