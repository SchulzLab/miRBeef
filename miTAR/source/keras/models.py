from tensorflow.keras.layers import Input, Embedding, Conv1D, Activation, MaxPooling1D, Concatenate, BatchNormalization, \
    Dropout
from tensorflow.keras.layers import Bidirectional, LSTM, GRU
from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1


# class ConvBlock(Model):
# 	def __init__(self,
# 	input_len,
# 	num_filters = 200,
# 	kernel_size = 12,
# 	dropout_rate = 0.2,
# 	):

# 	super(SequenceModel, self).__init__()

# 	def call(self, inputs):
# 		return

class SequenceModelEmbedding(Model):

    def __init__(self,
                 num_filters=24,
                 kernel_size=8,
                 dropout_rate=0.2,
                 ):
        super(SequenceModelEmbedding, self).__init__()
        # Keeping branches separated and assuming they might have different params
        # TODO Could use class "ConvBlock()"
        # l: left, r: right
        self.embedding_l = Embedding(input_dim=5, output_dim=5, input_length=26)
        # ValueError: Input 0 of layer "conv1d" is incompatible with the layer: expected min_ndim=3, found ndim=2. Full shape received: (27, 5)
        self.conv1_l = Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu')
        self.dropout1_l = Dropout(dropout_rate)
        self.maxpool_l = MaxPooling1D(pool_size=2)
        # self.dropout2_l = Dropout(dropout_rate)

        self.embedding_r = Embedding(input_dim=5, output_dim=5, input_length=53)
        self.conv1_r = Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu')
        self.dropout1_r = Dropout(dropout_rate)
        self.maxpool_r = MaxPooling1D(pool_size=2)
        # self.dropout2_r = Dropout(dropout_rate)

        self.concatenate = Concatenate(axis=1)

        self.bilstm = Bidirectional(LSTM(32, activation='relu'))

        self.dropout3 = Dropout(dropout_rate)
        self.fc1 = Dense(16, activation='relu')
        self.dropout4 = Dropout(dropout_rate)
        self.fc2 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        # print("embedding in: ", inputs[0].shape) # (26, 5)

        x_l = self.embedding_l(inputs[0])  # (None, 26, 5)

        # print("embedding out: ", x_l.shape)

        x_l = self.conv1_l(x_l)
        # print(x_l.shape)
        # x_l = self.conv1_l(tf.expand_dims(x_l, axis=-1))
        x_l = self.dropout1_l(x_l)
        x_l = self.maxpool_l(x_l)
        # x_l = self.dropout2_l(x_l)

        x_r = self.embedding_r(inputs[1])
        x_r = self.conv1_r(x_r)
        x_r = self.dropout1_r(x_r)
        x_r = self.maxpool_r(x_r)
        # x_r = self.dropout2_r(x_r)

        # print("concatenate in left: ", x_l.shape)
        # print("concatenate in right: ", x_r.shape)

        x = self.concatenate([x_l, x_r])

        # print("concatenate out: ", x.shape)

        x = self.bilstm(x)
        x = self.dropout3(x)
        x = self.fc1(x)
        x = self.dropout4(x)
        x = self.fc2(x)

        return x


class SequenceModelOneHot(Model):

    def __init__(self,
                 num_filters=12,
                 kernel_size=8,
                 dropout_rate=0.1,
                 rnn_units=32,
                 ):
        super(SequenceModelOneHot, self).__init__()
        # l: left, r: right
        self.conv1_l = Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu')
        self.dropout1_l = Dropout(rate=dropout_rate)
        self.maxpool_l = MaxPooling1D(pool_size=2)

        self.conv1_r = Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu')
        self.dropout1_r = Dropout(rate=dropout_rate)
        self.maxpool_r = MaxPooling1D(pool_size=2)

        self.concatenate = Concatenate(axis=1)

        # self.bilstm = Bidirectional(LSTM(units=rnn_units, activation='relu'))
        self.bilstm = Bidirectional(GRU(units=rnn_units, activation='relu'))

        self.dropout3 = Dropout(dropout_rate)
        self.fc1 = Dense(16, activation='relu')
        self.dropout4 = Dropout(dropout_rate)
        self.fc2 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        # print("input shape left: ", inputs[0].shape)
        # print("input shape right: ", inputs[1].shape)

        x_l = self.conv1_l(inputs[0])
        # print("conv1d left: ", x_l.shape)

        x_l = self.dropout1_l(x_l)
        x_l = self.maxpool_l(x_l)

        x_r = self.conv1_r(inputs[1])
        # print("conv1d right: ", x_l.shape)

        x_r = self.dropout1_r(x_r)
        x_r = self.maxpool_r(x_r)

        # print("concatenate in left: ", x_l.shape)
        # print("concatenate in right: ", x_r.shape)

        x = self.concatenate([x_l, x_r])
        # print("concatenate out: ", x.shape)

        x = self.bilstm(x)
        # print("lstm out: ", x.shape)

        x = self.dropout3(x)
        # print("dropout out: ", x.shape)

        x = self.fc1(x)
        # print("fc1 out: ", x.shape)

        x = self.dropout4(x)
        # print("dropout out: ", x.shape)

        x = self.fc2(x)
        # print("sigmoid out: ", x.shape)

        return x


class DMISOModel(Model):

    def __init__(self):
        super(DMISOModel, self).__init__()
        # First branch - mirna
        self.conv1_l = Conv1D(filters=10, kernel_size=8, name='conv1d_l_1')
        self.activation1_l = Activation('relu', name='activation_relu_l_1')
        self.maxpool_l = MaxPooling1D(pool_size=4, strides=1, padding='valid', name='max_pooling1d_l_1')
        # Second branch - target
        self.conv1_r = Conv1D(filters=10, kernel_size=8, name='conv1d_r_1')
        self.activation1_r = Activation('relu', name='activation_relu_r_1')
        self.maxpool_r = MaxPooling1D(pool_size=4, strides=1, padding='valid', name='max_pooling1d_r_1')

        self.concatenate = Concatenate(axis=1, name='concatenate_1')

        self.batchnorm1 = BatchNormalization(name='batch_normalization_1')
        self.dropout1 = Dropout(0.25, name='dropout_1')

        self.bilstm = Bidirectional(LSTM(10, return_sequences=True, name='lstm_1'), name='bidirectional_1')
        self.batchnorm2 = BatchNormalization(name='batch_normalization_2')
        self.dropout2 = Dropout(0.5, name='dropout_2')

        self.flatten = Flatten(name='flatten_1')
        self.dense1 = Dense(100, activation='linear', kernel_regularizer=l1(0.01), name='dense_1')
        self.batchnorm3 = BatchNormalization(name='batch_normalization_3')
        self.activation1 = Activation('relu', name='activation_relu_2')

        self.dropout3 = Dropout(0.5, name='dropout_3')
        self.dense2 = Dense(1, activation='linear', name='dense_2')
        self.batchnorm4 = BatchNormalization(name='batch_normalization_4')
        self.activation2 = Activation('sigmoid', name='activation_sigmoid_out')

    def call(self, inputs):
        input_mirna = inputs[0]  # (None, 30, 4)
        input_target = inputs[1]  # (None, 60, 4)

        x_l = self.conv1_l(input_mirna)  # (None, 23, 10)
        x_l = self.activation1_l(x_l)
        x_l = self.maxpool_l(x_l)  # (None, 20, 10)

        x_r = self.conv1_r(input_target)  # (None, 53, 10)
        x_r = self.activation1_r(x_r)
        x_r = self.maxpool_r(x_r)  # (None, 50, 10)

        x = self.concatenate([x_l, x_r])  # (None, 70, 10)

        x = self.batchnorm1(x)
        x = self.dropout1(x)

        x = self.bilstm(x)  # (None, 70, 20)
        x = self.batchnorm2(x)
        x = self.dropout2(x)

        x = self.flatten(x)  # (None, 1400)
        x = self.dense1(x)  # (None, 100)
        x = self.batchnorm3(x)
        x = self.activation1(x)

        x = self.dropout3(x)
        x = self.dense2(x)  # (None, 1)
        x = self.batchnorm4(x)
        x = self.activation2(x)

        return x
