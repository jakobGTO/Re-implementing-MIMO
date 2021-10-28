import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Conv2D, Add, Input, Flatten, AveragePooling2D, Reshape, Permute
from tensorflow.keras.datasets import cifar10, cifar100
from keras_multi_head import MultiHead

class MIMO_ResNet28_10(object):
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

        x_skip,x = inputs, inputs

        x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=filters,strides=(strides,strides), kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=filters, strides=(1,1), kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)

        if not x.shape.is_compatible_with(x_skip.shape):
            x_skip = Conv2D(filters=filters, strides=(strides,strides), kernel_size=(1,1), padding='same', use_bias=False, kernel_initializer='he_normal')(x_skip)

        x = Add()([x,x_skip])

        for i in range(3):
            x_skip,x = x,x

            x = BatchNormalization(momentum=0.99,epsilon=0.001)(x)
            x = Activation('relu')(x)
            x = Conv2D(filters=filters,strides=(1,1), kernel_size=(3,3), padding='same',use_bias=False, kernel_initializer='he_normal')(x)
            x = BatchNormalization(momentum=0.99,epsilon=0.001)(x)
            x = Activation('relu')(x)
            x = Conv2D(filters=filters,strides=(1,1), kernel_size=(3,3), padding='same',use_bias=False, kernel_initializer='he_normal')(x)

            if not x.shape.is_compatible_with(x_skip.shape):
                x_skip = Conv2D(filters=filters, strides=(strides,strides), kernel_size=(1,1), padding='same', use_bias=False, kernel_initializer='he_normal')(x_skip)

            x = Add()([x,x_skip])

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
        x = Permute([2,3,4,1])(inputs)
        x = Reshape(input_shape[1:-1] + [input_shape[-1] * M])(x)
        x = Conv2D(filters=16, strides=(1,1), kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)

        # We fit 4 resnet blocks to get a total network depth of 24
        # The filters are multiplied with 10 for the width multiplier
        x = self.block(x, filters=160, strides=1)
        x = self.block(x, filters=320, strides=2)
        x = self.block(x, filters=640, strides=2)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=8)(x)
        x = Flatten()(x)

        x = MultiHead(Dense(units=K, kernel_initializer='he_normal', activation=None), layer_num=M)(x)

        return Model(inputs=inputs, outputs=x)

    @tf.function
    def train_step(x_train, labels ):
        return None

    def fit():
        return None

    

if __name__ == '__main__':
    print(tf.__version__)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    mimo = MIMO_ResNet28_10()
    model = mimo.build(input_shape=(5,32,32,3),K=5,M=5)
    model.summary()
