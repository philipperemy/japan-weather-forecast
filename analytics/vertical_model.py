import keras.backend as K
from keras.layers import Dense, Lambda, Dropout, GRU
from keras.layers import Permute
from keras.models import Input, Model


class VerticalModel:

    def __init__(self, batch_size, param):
        """
        Vertical Model code
        :param batch_size: Batch size of the model. Integer. E.g 16.
        :param param: Some parameters required by the model. dropout_rate, hidden_size...
        """
        self.initial_epoch = 0
        inputs = Input(batch_shape=(batch_size, param.num_days, param.time_bins), name='Data')
        # mean = Input(batch_shape=(batch_size, 1, param.time_bins), name='Mean')
        # std = Input(batch_shape=(batch_size, 1, param.time_bins), name='Std')

        r_shape = K.shape(inputs)

        # (stock, day, time) -> (stock, time, day):
        x = Permute((2, 1))(inputs)

        # (stock, time, day) -> (stock * time, day, 1)
        x = Lambda(lambda y: K.reshape(y, (r_shape[0] * r_shape[2], r_shape[1], 1)))(x)

        # x = GRU(param.hidden_size, return_sequences=True)(x)
        from tcn import TCN
        x = TCN(nb_filters=param.hidden_size, dilations=[1, 2, 4, 8, 16, 32], return_sequences=True,
                 dropout_rate=param.dropout_rate)(x)

        # (stock * time, day, hidden_size) -> (stock, time, day, hidden_size)
        x = Lambda(lambda y: K.reshape(y, shape=(r_shape[0], r_shape[2], r_shape[1], param.hidden_size)))(x)

        x = Permute((2, 1, 3))(x)

        x = Dense(param.hidden_size)(x)
        x = Dropout(param.dropout_rate)(x)
        x = Dense(1, activation='linear')(x)  # could be positive, negative here.

        x = Lambda(lambda z_: K.squeeze(z_, axis=3))(x)

        # x = Lambda(lambda z_: K.exp(z_ * SCALE * std + mean) - 1e-2)(x)

        # self.model = Model(inputs=[inputs, mean, std], outputs=[x])
        self.model = Model(inputs=inputs, outputs=[x])

    def restore(self, checkpoint_path):
        """
        Restore the weights and the last epoch of a previous run.
        :param checkpoint_path: The path to the checkpoints.
        """
        from glob import glob
        from natsort import natsorted
        import os
        checkpoints = glob(checkpoint_path + '/**/*.h5', recursive=True)
        if len(checkpoints) > 0:
            last_checkpoint = natsorted(checkpoints)[-1]
            print(f'Loading weights from {last_checkpoint}.')
            self.model.load_weights(last_checkpoint)
            last_epoch = int(os.path.basename(last_checkpoint).split('_')[0])
            self.initial_epoch = last_epoch + 1
            print(f'Starting from epoch = {self.initial_epoch}.')
        else:
            print('No checkpoints were found. Going to train from scratch.')
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
