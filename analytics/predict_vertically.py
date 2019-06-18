# https://www.data.jma.go.jp/obd/stats/data/en/smp/index.html#remarks
import os
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

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

"""
BEST RUN WAS:
3744 41.46647466135122 52.765413584991336
"""


def ma(x, period=180):  # (num_stocks, num_days, num_bins)
    num_stocks, num_days, num_bins = x.shape
    assert len(x.shape) == 3
    x = np.roll(x, shift=1, axis=1)
    x = np.reshape(np.transpose(x, (0, 2, 1)), [num_stocks * num_bins, num_days])  # (num_stocks, num_bins, num_days)
    d = pd.DataFrame(np.transpose(x)).rolling(window=period, center=False).mean().fillna(method='bfill')
    a = np.transpose(d.values)
    a = np.transpose(np.reshape(a, [num_stocks, num_bins, num_days]), (0, 2, 1))
    return a


def mae(a, b):
    return np.mean(np.abs(a - b))


def norm(x):
    x = np.log(x + 1e-2)  # simple normalization. could be per station.
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)

    # import matplotlib.pyplot as plt
    # plt.hist(x.flatten(), 40)
    # plt.show()
    # Looks fairly gaussian.
    x = (x - mean) / std / SCALE

    return x, mean, std


def de_norm(x, mean, std):
    return np.exp(x * SCALE * std + mean) - 1e-2


def main():
    data_frames = []
    for input_filename in sorted(glob('../output/*_VIEW_13.json')):  # 13 -> monthly rainfall.
        d = read_file(input_filename)
        d.drop('Annual', axis=1, inplace=True)
        data_frames.append(d)

    # start at 1929.
    years_list = sorted(set(sum([list(d.index) for d in data_frames], [])))
    years_list = years_list[years_list.index('1929'):]
    # re index them back in 1929 or less.
    super_data_frames = []
    for d in data_frames:
        super_data_frames.append(d.reindex(years_list))

    # fill NaN values with the average mean.
    for d in super_data_frames:
        d.fillna(d.mean(), inplace=True)

    # correct.
    x = np.array([d.values for d in super_data_frames])
    x, mean, std = norm(x)
    y = np.roll(x, axis=1, shift=-1)

    # the last one does not make sense.
    x = x[:, :-1, :]
    y = y[:, :-1, :]

    assert x[0, 1, 0] == y[0, 0, 0]
    print(x.shape)
    train_until_index = x.shape[1] - HOW_MANY_YEARS_IN_TEST

    from vertical_model import VerticalModel

    class Params:
        num_days = None
        time_bins = 12
        hidden_size = 64
        dropout_rate = 0.2

    vm = VerticalModel(batch_size=None, param=Params)
    vm.model.compile(loss='mae', optimizer=Adam(lr=1e-4, clipnorm=1.))
    vm.restore('checkpoints')
    vm.model.summary()

    last_test_error = 1e9
    for step in range(int(1e5)):

        p = de_norm(vm.model.predict(x), mean, std)
        t = de_norm(y, mean, std)

        p_train = p[:, :train_until_index]
        p_test = p[:, train_until_index:]

        t_train = t[:, :train_until_index]
        t_test = t[:, train_until_index:]

        # print(t_train.shape, t_test.shape)
        ss_test_dumb = np.tile(np.median(t_train, axis=1, keepdims=True), (1, t_test.shape[1], 1))

        train_error = mae(p_train, t_train)
        test_error = mae(p_test, t_test)
        test_error_dumb = mae(ss_test_dumb, t_test)

        print(p_test[0, 0, :])
        print(t_test[0, 0, :])

        print(f'{str(step).zfill(5)}, tr = {train_error:.3f}, te = {test_error:.3f}, te_b = {test_error_dumb:.3f}')

        if test_error < last_test_error:
            for filename in glob('checkpoints/*.h5'):
                os.remove(filename)
            vm.model.save_weights(f'checkpoints/{step}_{train_error:.3f}_{test_error:.3f}.h5', overwrite=True)
            last_test_error = test_error

        vm.model.fit(x[:, :train_until_index], y[:, :train_until_index],
                     shuffle=True, batch_size=16, epochs=5, verbose=0)


if __name__ == '__main__':
    main()
