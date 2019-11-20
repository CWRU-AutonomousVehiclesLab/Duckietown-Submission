from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf


class FrankNet:
    @staticmethod
    def build_linear_branch(inputs=(100, 200, 3)):
        # ? Layer Normalization
        x = Lambda(lambda x: x/255.0)(inputs)

        # ? L1: CONV => RELU
        x = Conv2D(24, (5, 5), strides=(2, 2), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L2: CONV => RELU
        x = Conv2D(36, (5, 5), strides=(2, 2), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L3: CONV => RELU
        x = Conv2D(48, (5, 5), strides=(2, 2), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L4: CONV => RELU
        x = Conv2D(64, (3, 3), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L5: CONV => RELU
        x = Conv2D(64, (3, 3), padding="valid")(x)
        x = Activation("relu")(x)

        # ? Flatten
        x = Flatten()(x)

        # ? Fully Connected
        x = Dense(1164)(x)
        x = Dense(100)(x)
        x = Dense(50)(x)
        x = Dense(10)(x)
        x = Dense(1,name="Linear_Velocity_Out")(x)

        return x

    @staticmethod
    def build_angular_branch(inputs=(100, 200, 3)):
        # ? Layer Normalization
        x = Lambda(lambda x: x/255.0)(inputs)

        # ? L1: CONV => RELU
        x = Conv2D(24, (5, 5), strides=(2, 2), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L2: CONV => RELU
        x = Conv2D(36, (5, 5), strides=(2, 2), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L3: CONV => RELU
        x = Conv2D(48, (5, 5), strides=(2, 2), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L4: CONV => RELU
        x = Conv2D(64, (3, 3), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L5: CONV => RELU
        x = Conv2D(64, (3, 3), padding="valid")(x)
        x = Activation("relu")(x)

        # ? Flatten
        x = Flatten()(x)

        # ? Fully Connected
        x = Dense(1164)(x)
        x = Dense(100)(x)
        x = Dense(50)(x)
        x = Dense(10)(x)
        x = Dense(1,name="Angular_Velocity_Out")(x)

        return x

    @staticmethod
    def build(width=100, height=200):
        input_shape = (height, width, 3)
        inputs = Input(shape=input_shape)
        linearVelocity = FrankNet.build_linear_branch(inputs)
        angularVelocity = FrankNet.build_angular_branch(inputs)

        model = Model(inputs=inputs, outputs=[
            linearVelocity, angularVelocity], name="FrankNet")

        return model
