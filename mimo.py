from os import name
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Conv2D, Add, Input, Flatten, AveragePooling2D, Reshape, Permute
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import sparse_categorical_crossentropy
import numpy as np
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy, Mean
from robustness_metrics.metrics.uncertainty import ExpectedCalibrationError

loss_tracker = Mean(name="neg_likelihood")
accuracy = SparseCategoricalAccuracy(name="accuracy")
ece_tracker = ExpectedCalibrationError(num_bins=10)

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
    def __init__(self, mimomodel, num_batch_reps, M):
        super(MIMO, self).__init__()
        self.mimomodel = mimomodel
        self.num_batch_reps = num_batch_reps
        self.M = M

    def call(self, inputs, *args, **kwargs):
        return self.mimomodel(inputs)

    @tf.function
    def train_step(self, data):
        ''' Adapted from https://keras.io/guides/customizing_what_happens_in_fit/ '''
        x, y = data

        with tf.GradientTape() as tape:
            inputs, targets = batch_repetition(x,y,self.M,self.num_batch_reps,training=True)
            logits = self.mimomodel(inputs, training=True)
            trainable_vars = self.mimomodel.trainable_variables
            loss, loss_fn = custom_loss(targets, logits, trainable_vars)
        gradients = tape.gradient(loss_fn, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        loss_tracker.update_state(loss)
        accuracy.update_state(targets, logits)
        probs = tf.nn.softmax(logits)
        ece_tracker.add_batch(probs, label=targets)

        return {"neg_likelihood": loss_tracker.result(), "accuracy": accuracy.result(), "ece": ece_tracker.result()["ece"]}

    @tf.function
    def test_step(self, data):
        ''' Adapted from https://keras.io/guides/customizing_what_happens_in_fit/ '''
        x, y = data
        inputs, targets = batch_repetition(x,y,self.M,self.num_batch_reps,training=False)

        y_pred = self.mimomodel(inputs, training=False)
        y_pred = np.array(y_pred)
        targets = np.array(targets)

        # Calculate loss (already averaged across output nodes)
        loss = custom_loss(targets, y_pred, False)
        loss_tracker.update_state(loss)

        # Calculate average prediction across output nodes
        pred_sum = y_pred[:, 0, :].copy()
        for i in range(M):
            pred_sum += y_pred[:, i, :]
        pred_avg = pred_sum / M

        # Reshape targets average prediction
        sqz_targets = []
        for i in range(targets.shape[0]):
            sqz_targets.append(targets[i,0,0])
        sqz_targets = np.array(sqz_targets)[:,np.newaxis]
        
        # Calculate average accuracy
        accuracy.update_state(sqz_targets, pred_avg)

        # Calculate average ece
        probs = tf.nn.softmax(pred_avg)
        ece_tracker.add_batch(probs, label=sqz_targets)

        return {"neg_likelihood": loss_tracker.result(), "accuracy": accuracy.result(), "ece": ece_tracker.result()["ece"]}

    @property
    def metrics(self):
        return [loss_tracker, accuracy]


def custom_loss(y_true, y_pred, trainable_variables):
    ''' Loss function as described in Section 2 of original paper '''
    negative_likelihood = tf.reduce_mean(
                tf.reduce_sum(sparse_categorical_crossentropy(
                    y_true, y_pred, from_logits=True), axis=1))
    lossL2 = 0.0
    if trainable_variables:
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_variables]) * 3e-4
    
    return negative_likelihood, negative_likelihood+lossL2

def batch_repetition(x, y, ensemble_size, num_reps, training):
    xlist = []
    ylist = []

    for i in range(ensemble_size):
        if i == 0:
            xlist.append(x)
            ylist.append(y)
        else:
            if training:
                idx = tf.random.shuffle(tf.range(len(y)))
                xlist.append(tf.gather(x, idx))
                ylist.append(tf.gather(y, idx))
            else:
                xlist.append(x)
                ylist.append(y)

    if num_reps >= 1:
        inputs = tf.repeat(tf.stack(xlist, 1), repeats=num_reps, axis=0)
        targets = tf.repeat(tf.stack(ylist, 1), repeats=num_reps, axis=0)
    else:
        inputs = tf.stack(xlist, 1)
        targets = tf.stack(ylist, 1)

    return inputs, targets

if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)
    print(tf.__version__)

    M = 3
    K = 10
    batch_size = 32
    num_batch_reps = 0
    num_epochs = 10

    # Data handling
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    x_train = x_train[:50,:,:,:]
    y_train = y_train[:50,:]
    x_test = x_test[:50,:,:,:]
    y_test = y_test[:50,:]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # During training send in randomly sampled images
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    # During testing we send the same image to each node
    test_dataset = test_dataset.batch(batch_size)
    
    # Set optimizer, adapted from https://keras.io/api/optimizers/ and with
    # hyperparameters described in Annex B of the original paper.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2, decay_steps=100000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule, momentum=0.9, nesterov=True)

    # Define model and fit model
    input_shape=(M,32,32,3)

    resnet = ResNet20_10()
    resnet_architecture = resnet.build(input_shape, K, M)

    model = MIMO(resnet_architecture, num_batch_reps, M)
    model.compile(optimizer=optimizer)
    model.fit(train_dataset, batch_size=None, epochs=num_epochs, shuffle=True)
    results = model.evaluate(test_dataset, batch_size=None)

    '''
    # Get prediction for each subnetwork and average them
    pred_sum = preds[:, 0, :].copy()
    for i in range(1, M):
        pred_sum += preds[:, i, :]

    pred_avg = pred_sum / M

    print('Test loss: ', results)
    print('Print test acc: ',np.sum(pred_avg.argmax(axis=1) == y_test_oldshape.flatten()) / y_test_oldshape.shape[0])
    '''