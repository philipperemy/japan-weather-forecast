# https://www.data.jma.go.jp/obd/stats/data/en/smp/index.html#remarks
from glob import glob

import numpy as np
import pandas as pd
from keras.optimizers import Adam

from read_data import read_file

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

HOW_MANY_YEARS_IN_TEST = 10

SCALE = 10


def norm(x):
    x = np.log(x + 1e-6)  # simple normalization. could be per station.
    mean = np.mean(x)
    std = np.std(x)
    x = (x - mean) / std / SCALE
    return x, mean, std


def de_norm(x, mean, std):
    return np.exp(x * SCALE * std + mean) - 1e-6


def main():
    data_frames = []
    for input_filename in sorted(glob('../output/*_VIEW_13.json')):
        d = read_file(input_filename)
        d.drop('Annual', axis=1, inplace=True)
        data_frames.append(d)

    # re index them back in 1929 or less.
    super_index = sorted(set(sum([list(d.index) for d in data_frames], [])))
    super_data_frames = []
    for d in data_frames:
        super_data_frames.append(d.reindex(super_index))

    # fill NaN values with the average mean.
    for d in super_data_frames:
        d.fillna(d.mean(), inplace=True)

    # correct.
    x = np.array([d.values for d in super_data_frames])
    x, mean, std = norm(x)
    y = np.roll(x, axis=1, shift=-1)
    print(x.shape)
    train_until_index = x.shape[1] - HOW_MANY_YEARS_IN_TEST
    x = [x, np.ones(len(x)) * train_until_index]

    from vertical_model import VerticalModel

    class Params:
        num_days = None
        time_bins = 12
        hidden_size = 32
        dropout_rate = 0.2

    vm = VerticalModel(batch_size=256, param=Params)
    vm.model.compile(loss=vm.loss, optimizer=Adam(lr=1e-4, clipnorm=1.))

    for epoch in range(int(1e5)):

        p = de_norm(vm.model.predict(x), mean, std)
        t = de_norm(y, mean, std)

        p_train = p[:, :train_until_index]
        p_test = p[:, train_until_index:]

        t_train = t[:, :train_until_index]
        t_test = t[:, train_until_index:]

        train_error = np.mean(np.abs(p_train - t_train))
        test_error = np.mean(np.abs(p_test - t_test))
        # print(p.flatten()[0:20])
        # print(t.flatten()[0:20])
        # print('-' * 80)
        print(epoch, train_error, test_error)
        vm.model.fit(x, y, shuffle=True, epochs=5, batch_size=256, verbose=0)


if __name__ == '__main__':
    main()
