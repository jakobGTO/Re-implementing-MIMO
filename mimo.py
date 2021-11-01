import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Conv2D, Add, Input, Flatten, AveragePooling2D, Reshape, Permute
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import sparse_categorical_crossentropy
import numpy as np

class CustomLayer(Dense):
    """Adapted from https://keras.io/guides/making_new_layers_and_models_via_subclassing/"""
    def __init__(self, units, M, kernel_initializer='he_normal'):
        super().__init__(units=units, kernel_initializer=kernel_initializer)
        self.M = M

    def call(self, inputs):
        x = super().call(inputs)
        x = tf.reshape(x, (tf.shape(inputs)[0], self.M, self.units // self.M))
        return x

class MIMO_ResNet28_10(Model):
    def __init__(self):
        super(MIMO_ResNet28_10, self).__init__()

    def block(self, inputs, filters,  strides):
        """
        https://paperswithcode.com/method/residual-block
        Build ResNet-28-10 Blocks and group them
        args:
            inputs: tf.tensor. 2D Conv layer
            filters: int. Number of filters in the 2D Conv Layer
            strides: int. Number of strides in the 2D Conv Layer
            n_blocks: int. Number of blocks to group the ResNet blocks into
        returns:
            tf.tensor
        """

        x_skip, x = inputs, inputs

        x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=filters, strides=(strides, strides), kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=filters, strides=(1, 1), kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)

        x_skip = Conv2D(filters=filters, strides=(strides, strides), kernel_size=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(x_skip)

        x = Add()([x, x_skip])

        for i in range(3):
            x_skip, x = x, x

            x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
            x = Activation('relu')(x)
            x = Conv2D(filters=filters, strides=(1,1), kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
            x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
            x = Activation('relu')(x)
            x = Conv2D(filters=filters, strides=(1, 1), kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)

            x = Add()([x, x_skip])

        return x

    def build(self, input_shape, K, M):
        """
        https://arxiv.org/pdf/1605.07146.pdf
        https://paperswithcode.com/paper/wide-residual-networks
        Build ResNet-28-10 Model
        args:
            input_shape: tf.tensor. Input shape (M, img width, img height, channels)
            K: int. number of classes of the dataset 
            M: int. size of ensemble
        returns:
            tf.keras.Model
        """

        input_shape = list(input_shape)
        inputs = Input(shape=input_shape)
        # dim_1 -> dim_2, dim_2 -> dim_3, dim_3 -> dim_4, dim_4 -> dim_1
        # where dim_1 = size of ensemble, dim_2 = width, dim_3 = heigh, dim_4 = channels
        x = Permute([2, 3, 4, 1])(inputs)
        x = Reshape(input_shape[1:-1] + [input_shape[-1] * M])(x)
        x = Conv2D(filters=16, strides=(1, 1), kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)

        # The filters are multiplied with 10 for the width multiplier
        x = self.block(x, filters=160, strides=1)
        x = self.block(x, filters=320, strides=2)
        x = self.block(x, filters=640, strides=2)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=8)(x)
        x = Flatten()(x)

        x = CustomLayer(units=K*M, kernel_initializer='he_normal', M=M)(x)

        return Model(inputs=inputs, outputs=x)

    @tf.function
    def train_step(self, x, y):
        ''' Adapted from https://keras.io/guides/customizing_what_happens_in_fit/ '''
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            trainable_vars = model.trainable_variables
            loss = custom_loss(y, logits)
            
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        ''' Adapted from https://keras.io/guides/customizing_what_happens_in_fit/ '''
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def custom_loss(y_true, y_pred):
    ''' Loss function as described in Section 2 of original paper '''
    negative_likelihood = tf.reduce_mean(
                tf.reduce_sum(sparse_categorical_crossentropy(
                    y_true, y_pred, from_logits=True), axis=1))
    #lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_variables if 'bias' not in v.name]) * 3e-4
    return negative_likelihood #+ lossL2

if __name__ == '__main__':
    print(tf.__version__)

    # Data handling
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x = np.repeat(x_train[:,np.newaxis,:,:,:], 5, axis=1)
    y = np.repeat(y_train[:,np.newaxis,:], 5, axis=1)


    # Set optimizer, adapted from https://keras.io/api/optimizers/ and with
    # hyperparameters described in Annex B of the original paper.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule, nesterov=True)


    mimo = MIMO_ResNet28_10()
    model = mimo.build(input_shape=(5, 32, 32, 3), K=10, M=5)

    model.compile(optimizer=optimizer, loss=custom_loss)
    model.fit(x, y, batch_size=24, epochs=10)
    #model.evaluate(x_test, y_test)
