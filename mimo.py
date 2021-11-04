import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Conv2D, Add, Input, Flatten, AveragePooling2D, Reshape, Permute
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import sparse_categorical_crossentropy
import numpy as np
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy, Mean
from robustness_metrics.metrics import ExpectedCalibrationError

loss_tracker = Mean(name="neg_likelihood")
accuracy = SparseCategoricalAccuracy(name="accuracy")
ece_tracker = ExpectedCalibrationError()

class CustomLayer(Dense):
    """Adapted from https://keras.io/guides/making_new_layers_and_models_via_subclassing/"""
    def __init__(self, units, M, kernel_initializer='he_normal'):
        super().__init__(units=units, kernel_initializer=kernel_initializer)
        self.M = M

    def call(self, inputs):
        x = super().call(inputs)
        x = tf.reshape(x, (tf.shape(inputs)[0], self.M, self.units // self.M))
        return x

class ResNet20_10():
    def __init__(self):
        pass
        
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

class MIMO(Model):
    def __init__(self, mimomodel):
        super(MIMO, self).__init__()
        self.mimomodel = mimomodel

    @tf.function
    def train_step(self, data):
        ''' Adapted from https://keras.io/guides/customizing_what_happens_in_fit/ '''
        x, y = data
        with tf.GradientTape() as tape:
            logits = self.mimomodel(x, training=True)
            trainable_vars = self.mimomodel.trainable_variables
            loss_fn = custom_loss(y, logits, trainable_vars)
        gradients = tape.gradient(loss_fn, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        loss_tracker.update_state(loss_fn)
        accuracy.update_state(y, logits)
        ece_tracker.add_batch(logits, label=y)
        return {"neg_likelihood": loss_tracker.result(), "accuracy": accuracy.result(), "ece": ece_tracker.result()["ece"]}

    @tf.function
    def test_step(self, data):
        ''' Adapted from https://keras.io/guides/customizing_what_happens_in_fit/ '''
        x, y = data
        y_pred = self.mimomodel(x, training=False)
        loss = custom_loss(y, y_pred)
        loss_tracker.update_state(loss)
        accuracy.update_state(y, y_pred)
        ece_tracker.add_batch(y_pred, label=y)
        return {"neg_likelihood": loss_tracker.result(), "accuracy": accuracy.result(), "ece": ece_tracker.result()["ece"]}

    # implement the call method
    def call(self, inputs, *args, **kwargs):
        return self.mimomodel(inputs)


def custom_loss(y_true, y_pred, trainable_variables=None):
    ''' Loss function as described in Section 2 of original paper '''
    negative_likelihood = tf.reduce_mean(
                tf.reduce_sum(sparse_categorical_crossentropy(
                    y_true, y_pred, from_logits=True), axis=1))
    if trainable_variables:
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_variables if 'bias' not in v.name]) * 3e-4
        loss = negative_likelihood + lossL2
    else:
        loss = negative_likelihood
    return loss

if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)
    print(tf.__version__)

    # Data handling
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    M = 3

    # At train time pass random image through each node
    x_train = np.repeat(x_train[:,np.newaxis,:,:,:], M, axis=1)
    y_train = np.repeat(y_train[:,np.newaxis,:], M, axis=1)

    idx = np.arange(x_train.shape[0])
    shuffled_idx = np.random.permutation(x_train.shape[0])
    x_train = x_train[shuffled_idx,:,:,:,:] 
    y_train = y_train[shuffled_idx,:,:]

    # At test time pass same image through all input nodes
    y_test_oldshape = y_test
    x_test = np.repeat(x_test[:,np.newaxis,:,:,:], M, axis=1)
    y_test = np.repeat(y_test[:,np.newaxis,:], M, axis=1)

    
    # Set optimizer, adapted from https://keras.io/api/optimizers/ and with
    # hyperparameters described in Annex B of the original paper.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule, nesterov=True)

    # Define model and fit model
    K = 10
    input_shape=(x_train.shape[1:])

    resnet = ResNet20_10()
    resnet_architecture = resnet.build(input_shape, K, M)

    model = MIMO(resnet_architecture)
    model.compile(optimizer=optimizer, metrics=[loss_tracker, accuracy, ece_tracker])
    model.fit(x_train, y_train, batch_size=16, epochs=33, shuffle=True)
    results = model.evaluate(x_test, y_test, batch_size=16)
    preds = model.predict(x_test)

    # Get prediction for each subnetwork and average them
    pred_sum = preds[:, 0, :].copy()
    for i in range(1, M):
        pred_sum += preds[:, i, :]

    pred_avg = pred_sum / M

    print('Test loss: ', results)
    print('Print test acc: ',np.sum(pred_avg.argmax(axis=1) == y_test_oldshape.flatten()) / y_test_oldshape.shape[0])
