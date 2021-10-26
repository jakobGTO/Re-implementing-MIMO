import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D, Add, Input, Flatten, AveragePooling2D, Reshape, Permute

class mimo_resnet2810():
    def __init__(self):
        pass

    def resnet_block(inputs, filters,  strides, n_blocks):
        """
        https://paperswithcode.com/method/residual-block

        Build ResNet-28-10 Blocks and group them

        args:
            inputs: tf.Tensor. 2D Conv layer
            filters: int. Number of filters in the 2D Conv Layer
            strides: int. Number of strides in the 2D Conv Layer
            n_blocks: int. Number of blocks to group the ResNet blocks into
        returns:
            tf.Tensor
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

    def build_resnet(input_shape, K, M):
        """
        https://arxiv.org/pdf/1605.07146.pdf
        https://paperswithcode.com/paper/wide-residual-networks

        Build ResNet-28-10 Model

        args:
            input_shape: tf.Tensor. Input shape (M, w, h, channels)
            K: int. number of classes of the dataset 
            M: int. size of ensemble
        returns:
            tf.keras.Model
        """
        inputs = Input(shape=list(input_shape))




if __name__ == '__main__':
    print(tf.__version__)

