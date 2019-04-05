# https://www.data.jma.go.jp/obd/stats/data/en/smp/index.html#remarks
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dense

from analytics.read_data import read_file

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)


def main():
    d = read_file('../output/STATION_47401_VIEW_13.json')
    d.drop('Annual', axis=1, inplace=True)
    print(d)
    time_sequence = d.values.flatten()
    time_sequence = time_sequence[~np.isnan(time_sequence)]
    time_sequence = np.log(time_sequence)  # simple normalization.
    lookback_window = 50
    sequences = []
    targets = []
    for i in range(lookback_window, len(time_sequence) - lookback_window):
        sequences.append(time_sequence[i - lookback_window:i])
        targets.append(time_sequence[i])
    sequences = np.expand_dims(sequences, axis=-1)
    targets = np.expand_dims(targets, axis=-1)

    print(sequences.shape)
    print(targets.shape)

    m = Sequential()
    m.add(LSTM(32, input_shape=sequences.shape[1:]))
    m.add(Dense(1, activation='relu'))
    m.compile(loss='mse', optimizer='rmsprop')
    # validation_split: Float between 0 and 1.
    #                 Fraction of the training data to be used as validation data.
    #                 The model will set apart this fraction of the training data,
    #                 will not train on it, and will evaluate
    #                 the loss and any model metrics
    #                 on this data at the end of each epoch.
    #                 The validation data is selected from the last samples
    #                 in the `x` and `y` data provided, before shuffling.
    for epoch in range(100):
        m.fit(sequences, targets, shuffle=False, validation_split=0.2)
        error = np.mean(np.abs(np.exp(m.predict(sequences)) - np.exp(targets)))
        print(f'mean prediction = {np.mean(np.exp(m.predict(sequences)))}.')
        print(f'Average unit error: {error}.')


if __name__ == '__main__':
    main()
