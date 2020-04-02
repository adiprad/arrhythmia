import wfdb # used to read mitdb records
# from wfdb import processing # xqrs detector
import h5py # used to create database
import numpy as np # used for data manipulation

from keras.models import Sequential, load_model
from keras.layers import Dense
# from keras.utils import to_categorical
import matplotlib.pyplot as plt
from random import randint

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # suppress deprecation msgs
# from pprint import pprint
# from collections import Counter

# 'N': 21350, 'V': 3210, 'L': 2491, 'R': 1825, 'F': 759, '+': 329, '~': 112, 'A': 97, 'a': 31, '|': 18, 'S': 2, 'Q': 2, 'E': 1

# list of record files in mitdb by number
records   = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]

# creates an HDF5 dataset from mitdb, since reading every single record is expensive
def create_dataset_from_records(records):
    norm = [] # stores normal heartbeat data
    pvc  = [] # stores PVC heartbeat data
    lbbb = [] # stores LBBB heartbeat data
    rbbb = [] # stores RBBB heartbeat data

    for record in records:
        signals, fields = wfdb.rdsamp('mitdb/' + str(record)) # read record signals
        annotations     = wfdb.rdann('mitdb/' + str(record), 'atr') # read record annotations
        window_size     = 90 # size of window for each sample

        MLII = [sig[0] for sig in signals] # using MLII data

        # slice off data that doesn't fit in the window
        start = 0
        while annotations.sample[start] < window_size:
            start += 1

        end = 0
        while fields['sig_len'] - annotations.sample[end - 1] < window_size:
            end -= 1

        annos = zip(annotations.sample[start : end], annotations.symbol[start : end])

        # extracts normal and pvc data
        for sample, symbol in annos:
            if symbol == 'N':
                norm.append(MLII[sample - window_size : sample + window_size])
            elif symbol == 'V':
                pvc.append(MLII[sample - window_size : sample + window_size])
            elif symbol == 'L':
                lbbb.append(MLII[sample - window_size : sample + window_size])
            elif symbol == 'R':
                rbbb.append(MLII[sample - window_size : sample + window_size])

    # write to HDF5
    min_len = len(rbbb)
    with h5py.File('mitdb.hdf5', 'w') as f:
        f.create_dataset('normal', data=np.array(norm)[:min_len])
        f.create_dataset('pvc', data=np.array(pvc)[:min_len])
        f.create_dataset('lbbb', data=np.array(lbbb)[:min_len])
        f.create_dataset('rbbb', data=np.array(rbbb))
    return

# create_dataset_from_records(records)

# print(len(dataset['normal']))
# print(len(dataset['pvc']))
# print(len(dataset['lbbb']))
# print(len(dataset['rbbb']))
#
# beat = randint(1,1000)
#
# plt.plot(dataset['normal'][beat])
# plt.ylabel('Normal beat #{}'.format(beat))
# plt.show()
#
# plt.plot(dataset['pvc'][beat])
# plt.ylabel('Premature ventricular contraction beat #{}'.format(beat))
# plt.show()
#
# plt.plot(dataset['lbbb'][beat])
# plt.ylabel('Left bundle branch block beat #{}'.format(beat))
# plt.show()

# normal_detector = load_model('models/normal.h5')
# pvc_detector    = load_model('models/pvc.h5')
# lbbb_detector   = load_model('models/lbbb.h5')
#
# dataset = h5py.File('mitdb.hdf5', 'r')
#
# n_pred = lbbb_detector.evaluate(dataset['normal'], np.ones(21350))
# p_pred = lbbb_detector.evaluate(dataset['pvc'], np.ones(3210))
# l_pred = lbbb_detector.evaluate(dataset['lbbb'], np.ones(2491))
#
# print(n_pred)
# print(p_pred)
# print(l_pred)

def train_model(name, label):
    model = Sequential()

    model.add(Dense(30, activation='relu', input_dim=180))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    dataset = h5py.File('mitdb.hdf5', 'r')
    # interleave data for even training
    # print(dataset['normal'].shape)

    x = np.empty((7300, 180))
    x[0::4] = dataset['normal']
    x[1::4] = dataset['pvc']
    x[2::4] = dataset['lbbb']
    x[3::4] = dataset['rbbb']

    # generate labels
    y = np.tile(label, 1825)
    # print(x.shape)
    # print(y.shape)

    # moment of truth
    model.fit(
        x=x, y=y, batch_size=64,
        verbose=2, validation_split=0.2
    )

    dataset.close()
    model.save(f'models/{name}.h5')
    del model

# text formatting for readability
print('\033[94m\033[1m\033[4m \u2193 NORMAL MODEL \u2193 \033[24m\033[91m')
train_model('normal', np.array([1, 0, 0, 0]))

print('\033[94m\033[1m\033[4m \u2193 PVC MODEL \u2193 \033[24m\033[91m')
train_model('pvc',    np.array([0, 1, 0, 0]))

print('\033[94m\033[1m\033[4m \u2193 LBBB MODEL \u2193 \033[24m\033[91m')
train_model('lbbb',   np.array([0, 0, 1, 0]))

print('\033[94m\033[1m\033[4m \u2193 RBBB MODEL \u2193 \033[24m\033[91m')
train_model('rbbb',   np.array([0, 0, 0, 1]))

print('\033[0m\033[2m')

with h5py.File('mitdb.hdf5', 'r') as dataset:
    norm_beat = randint(1,1000)
    pvc_beat = randint(1,1000)
    lbbb_beat = randint(1,1000)
    rbbb_beat = randint(1,1000)

    plt.figure(figsize=[9, 6.75])
    plt.subplot(221)
    plt.plot(dataset['normal'][norm_beat])
    plt.title(f'Normal heartbeat #{norm_beat}')
    plt.subplot(222)
    plt.plot(dataset['pvc'][pvc_beat])
    plt.title(f'PVC heartbeat #{pvc_beat}')
    plt.subplot(223)
    plt.plot(dataset['lbbb'][lbbb_beat])
    plt.title(f'LBBB heartbeat #{lbbb_beat}')
    plt.subplot(224)
    plt.plot(dataset['rbbb'][rbbb_beat])
    plt.title(f'RBBB heartbeat #{rbbb_beat}')

    plt.show()

print('\033[0m')
