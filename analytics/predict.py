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


def main():
    sequences = []
    targets = []
    lookback_window = 50
    for input_filename in glob('../output/*_VIEW_13.json'):
        # print(input_filename)
        d = read_file(input_filename)
        d.drop('Annual', axis=1, inplace=True)
        # print(d)
        time_sequence = d.values.flatten()
        time_sequence = time_sequence[~np.isnan(time_sequence)]
        # normalization
        time_sequence = np.log(time_sequence + 1e-6)  # simple normalization. could be per station.
        time_sequence = (time_sequence - np.mean(time_sequence)) / np.std(time_sequence)
        for i in range(lookback_window, len(time_sequence) - lookback_window):
            sequences.append(time_sequence[i - lookback_window:i])
            targets.append(time_sequence[i])

    sequences = np.expand_dims(sequences, axis=-1)
    targets = np.expand_dims(targets, axis=-1)
    print(sequences.shape)
    print(targets.shape)

    # print(np.isnan(sequences).any())
    # print(np.max(sequences))
    # print(np.min(sequences))

    m = Sequential()
    m.add(LSTM(128, input_shape=sequences.shape[1:]))
    m.add(Dense(128, activation='relu'))
    m.add(Dense(1, activation='linear'))
    opt = RMSprop(lr=1e-5)
    m.compile(loss='mae', optimizer=opt)
    # validation_split: Float between 0 and 1.
    #                 Fraction of the training data to be used as validation data.
    #                 The model will set apart this fraction of the training data,
    #                 will not train on it, and will evaluate
    #                 the loss and any model metrics
    #                 on this data at the end of each epoch.
    #                 The validation data is selected from the last samples
    #                 in the `x` and `y` data provided, before shuffling.
    m.fit(sequences, targets, shuffle=True, validation_split=0.2, epochs=100, batch_size=256)
    # for epoch in range(100):
    #     error = np.mean(np.abs(np.exp(m.predict(sequences)) - np.exp(targets)))
    #     print(f'mean prediction = {np.mean(np.exp(m.predict(sequences)))}.')
    #     print(f'Average unit error: {error}.')


if __name__ == '__main__':
    main()
