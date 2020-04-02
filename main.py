import wfdb # used to read mitdb records
import h5py # used to create database
import numpy as np # used for data manipulation

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
# from keras.utils import to_categorical
# import matplotlib.pyplot as plt
# from random import randint
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

    # write to HDF5
    with h5py.File('mitdb.hdf5', 'w') as f:
        f.create_dataset('normal', data=np.array(norm))
        f.create_dataset('pvc', data=np.array(pvc))
        f.create_dataset('lbbb', data=np.array(lbbb))
    return

# create_dataset_from_records(records)
dataset = h5py.File('mitdb.hdf5', 'r')

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

# 80/20 percent split of data into training and testing (3210 samples * 0.8 = 2568 samples)
length = dataset['normal'].shape[0]
train_size  = int(length * 0.8)
test_size   = length - train_size

# split data into training and testing (80/20 % split)
x_train = np.vstack( (dataset['normal'][:train_size], dataset['pvc'][:train_size], dataset['lbbb'][:train_size]) )
x_test  = np.vstack( (dataset['normal'][train_size:], dataset['pvc'][train_size:], dataset['lbbb'][train_size:]) )

y_train = to_categorical( np.concatenate( (np.full(train_size, 0), np.full(train_size, 1), np.full(train_size, 2)) ) )
y_test  = to_categorical( np.concatenate( (np.full(test_size, 0), np.full(test_size, 1), np.full(test_size, 2)) ) )

dataset.close()

model = Sequential()

model.add(Dense(90, activation='relu', input_dim=180))
model.add(Dense(30, activation='relu', input_dim=180))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

save = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1)

model.fit(
    x=x_train, y=y_train,
    batch_size=64, epochs=3,
    verbose=2, callbacks=[save],
    validation_split=0, validation_data=(x_test, y_test),
    shuffle=True
)

# peak detection for choosing window for input
# threshold = 0.3
# signal_bool = False
# max_sample = 0
# peaks = []
#
# for c, sig in enumerate(MLII):
#     if sig > threshold and signal_bool == False:
#         signal_bool = True
#         max_sample = c
#     if sig < threshold and signal_bool == True :
#         signal_bool = False
#         peaks.append(max_sample)
#
#     if signal_bool == True and sig > MLII[max_sample]:
#         max_sample = c
# peak detection
