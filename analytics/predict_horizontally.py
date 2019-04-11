# https://www.data.jma.go.jp/obd/stats/data/en/smp/index.html#remarks
from glob import glob

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop

from read_data import read_file

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

HOW_MANY_YEARS_IN_TEST = 10


def main():
    lookback_window = 50
    raw_time_sequences = []
    cutoffs = []
    for input_filename in sorted(glob('../output/*_VIEW_13.json'))[0:20]:
        d = read_file(input_filename)
        d.drop('Annual', axis=1, inplace=True)
        d_train = d.head(len(d) - HOW_MANY_YEARS_IN_TEST)
        cutoff_train_test_index = len(d_train.values.flatten())
        time_sequence = d.values.flatten()
        time_sequence = time_sequence[~np.isnan(time_sequence)]
        time_sequence = np.log(time_sequence + 1e-6)  # simple normalization. could be per station.
        raw_time_sequences.append(time_sequence)
        cutoffs.append(cutoff_train_test_index)
    mean = np.mean(np.concatenate(raw_time_sequences))
    std = np.std(np.concatenate(raw_time_sequences))
    scale = 10

    x_train, y_train, x_test, y_test = [], [], [], []
    for time_sequence, cutoff_train_test_index in zip(raw_time_sequences, cutoffs):
        # normalization
        time_sequence = (time_sequence - mean) / std / scale
        for i in range(lookback_window, len(time_sequence)):
            model_input_slice = time_sequence[i - lookback_window:i]
            if i < cutoff_train_test_index:
                x_train.append(model_input_slice)
                y_train.append(time_sequence[i])
            else:
                x_test.append(model_input_slice)
                y_test.append(time_sequence[i])

    x_train = np.expand_dims(x_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    m = Sequential()
    m.add(LSTM(128, input_shape=x_train.shape[1:]))
    m.add(Dense(128, activation='relu'))
    m.add(Dense(1, activation='linear'))
    opt = RMSprop(lr=1e-4)
    m.compile(loss='mae', optimizer=opt)

    for epoch in range(100):
        p = np.exp(m.predict(x_test) * scale * std + mean)
        t = np.exp(y_test * scale * std + mean)
        error = np.mean(np.abs(p - t))
        print(epoch, error)
        m.fit(x_train, y_train, shuffle=True, validation_data=(x_test, y_test), epochs=1, batch_size=256, verbose=0)


if __name__ == '__main__':
    main()
