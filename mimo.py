import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D, Add, Input, Flatten, AveragePooling2D, Reshape, Permute

class mimo_resnet2810():
    def __init__(self):
        pass

    def resnet_block(self, inputs, filters,  strides, n_blocks):
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

        x = inputs
        y = inputs

        y = BatchNormalization(momentum=0.99, epsilon=0.001)
        y = Activation('relu')(y)
        y = Conv2D(filters=filters,strides=strides, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal')
        y = BatchNormalization(momentum=0.99, epsilon=0.001)
        y = Activation('relu')(y)
        y = Conv2D(filters=filters, strides=1, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal')

        x = Add([x,y])

        for i in range(n_blocks-1):
            y = x

            y = BatchNormalization(momentum=0.99,epsilon=0.001)
            y = Activation('relu')(y)
            y = Conv2D(filters=filters,strides=1, kernel_size=3, padding='same',use_bias=False, kernel_initializer='he_normal')
            y = BatchNormalization(momentum=0.99,epsilon=0.001)
            y = Activation('relu')(y)
            y = Conv2D(filters=filters,strides=1, kernel_size=3, padding='same',use_bias=False, kernel_initializer='he_normal')

            x = Add([x,y])

        return x

    def build_resnet(self, input_shape, K, M):
        """
        https://arxiv.org/pdf/1605.07146.pdf
        https://paperswithcode.com/paper/wide-residual-networks

        Build ResNet-28-10 Model

        args:
            input_shape: tf.tensor. Input shape (M, w, h, channels)
            K: int. number of classes of the dataset 
            M: int. size of ensemble
        returns:
            tf.keras.Model
        """

        inputs = Input(shape=list(input_shape))
        # dim_1 -> dim_2, dim_2 -> dim_3, dim_3 -> dim_4, dim_4 -> dim_1
        # where dim_1 = size of ensemble, dim_2 = width, dim_3 = heigh, dim_4 = channels
        x = Permute([2,3,4,1])(inputs)
        x = Reshape(input_shape[1:-1] + input_shape[-1] * M)(x)
        x = Conv2D(filters=16, strides=1)(x)

        # We fit 4 resnet blocks to get a total network depth of 24
        # The filters are multiplied with 10 for the width multiplier
        x = self.resnet_block(x, filters=160, strides=1, n_blocks=4)
        x = self.resnet_block(x, filters=320, strides=2, n_blocks=4)
        x = self.resnet_block(x, filters=640, strides=2, n_blocks=4)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        

if __name__ == '__main__':
    print(tf.__version__)

