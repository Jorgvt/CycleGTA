import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import keras


from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import NonNeg

__all__ = ["build_unet"]

class GDN(Layer):
    def __init__(self, 
                 filter_shape = (3,3), 
                 **kwargs):
      
        self.filter_shape = filter_shape

        super(GDN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(name = 'beta', 
                                    shape = (input_shape.as_list()[-1]),
                                    initializer = tf.keras.initializers.constant(1.0),
                                    trainable = True,
                                    constraint = lambda x: tf.clip_by_value(x, 1e-15, np.inf))
        
        self.alpha = self.add_weight(name = 'alpha', 
                                     shape = (input_shape.as_list()[-1]),
                                     initializer = tf.keras.initializers.constant(1.0),
                                     trainable = False,
                                     constraint = NonNeg())

        self.epsilon = self.add_weight(name = 'epsilon', 
                                     shape = (input_shape.as_list()[-1]),
                                     initializer = tf.keras.initializers.constant(1.0),
                                     trainable = False,
                                     constraint = NonNeg())
        
        self.gamma = self.add_weight(name = 'gamma', 
                                     shape = (self.filter_shape[0], self.filter_shape[1], input_shape.as_list()[-1], input_shape.as_list()[-1]),
                                     initializer = tf.keras.initializers.Zeros,
                                     trainable = True,
                                     constraint = NonNeg())
        
        
        super(GDN, self).build(input_shape)

    def call(self, x):
        norm_conv2 = tf.nn.convolution(tf.abs(x)**self.alpha,
                                      self.gamma,
                                      strides = (1, 1),
                                      padding = "SAME",
                                      data_format = "NHWC")

        norm_conv = self.beta + norm_conv2
        norm_conv = norm_conv**self.epsilon
        return x / norm_conv
        
    def compute_output_shape(self, input_shape):
        return (input_shape, self.output_dim)

def conv_block(inputs, filters, pool = True):
    ''' Define a convolutional block of the encoder.
  
    Arguments:
        - inputs (tf tensor): Input tensor
        - filters (int): Number of filters
        - pool (bool): Do the MaxPooling or not -- True

    Returns:
        - x (tf tensor): Output tensor before the MaxPooling
        - p (tf tensor): Output tensor after the MaxPooling '''

    x = Conv2D(filters, 3, padding = "same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Conv2D(filters, 3, padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    if pool == True:
        p = MaxPool2D((2, 2))(x)
        return x, p
    else:
        return x

def build_unet(shape=(96,256,3), num_classes=30, path_weights=None):
    ''' Build an UNET form model with GDN.
  
    Arguments:
        - shape ((int, int, int)): Shape of the input tensor
        - num_classes (int): Number of classes

    Returns:
        - model (tf model): Model built '''

    inputs = Input(shape)

    # Encoder
    g1 = GDN()(inputs)
    x1, p1 = conv_block(g1, 16, pool = True)
    g2 = GDN()(p1)
    x2, p2 = conv_block(g2, 32, pool = True)
    g3 = GDN()(p2)
    x3, p3 = conv_block(g3, 64, pool = True)
    g4 = GDN()(p3)
    x4, p4 = conv_block(g4, 128, pool = True)

    # Bridge
    b1 = conv_block(p4, 256, pool = False)

    # Decoder
    #u1 = UpSampling2D((2, 2), interpolation = "bilinear")(b1)
    u1 = Conv2DTranspose(256, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(b1)
    c1 = Concatenate()([u1, x4])
    x5 = conv_block(c1, 128, pool = False)

    #u2 = UpSampling2D((2, 2), interpolation = "bilinear")(x5)
    u2 = Conv2DTranspose(128, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(x5)
    c2 = Concatenate()([u2, x3])
    x6 = conv_block(c2, 64, pool = False)

    #u3 = UpSampling2D((2, 2), interpolation = "bilinear")(x6)
    u3 = Conv2DTranspose(64, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(x6)
    c3 = Concatenate()([u3, x2])
    x7 = conv_block(c3, 32, pool = False)

    #u4 = UpSampling2D((2, 2), interpolation = "bilinear")(x7)
    u4 = Conv2DTranspose(32, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(x7)
    c4 = Concatenate()([u4, x1])
    x8 = conv_block(c4, 16, pool = False)

    # Output
    output = Conv2D(num_classes, 1, padding = 'same', activation = 'softmax')(x8)

    # Model
    model = Model(inputs, output)

    # Load weights
    if path_weights is not None: model.load_weights(path_weights)

    return model

