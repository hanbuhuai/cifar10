# -*- coding UTF-8 -*-
import tensorflow as tf
'''
   vggnet 网络搭建
'''
def convnet(inputs, activation, kernel_initializer):
    conv1_1 = tf.layers.conv2d(inputs,32,(3,3),
        padding='same',
        activation=activation,
        kernel_initializer=kernel_initializer,
        name = 'conv1_1'
    )
    conv1_2 = tf.layers.conv2d(conv1_1,32,(3,3),
        padding='same',
        activation=activation,
        kernel_initializer=kernel_initializer,
        name = 'conv1_2'
    )
    pooling1 = tf.layers.max_pooling2d(conv1_2,
                                       (2, 2), # kernel size
                                       (2, 2), # stride
                                       name = 'pool1')
    conv2_1 = tf.layers.conv2d(pooling1,
                               32, # output channel number
                               (3,3), # kernel size
                               padding = 'same',
                               activation = activation,
                               kernel_initializer = kernel_initializer,
                               name = 'conv2_1')
    conv2_2 = tf.layers.conv2d(conv2_1,
                               32, # output channel number
                               (3,3), # kernel size
                               padding = 'same',
                               activation = activation,
                               kernel_initializer = kernel_initializer,
                               name = 'conv2_2')
    # 8 * 8
    pooling2 = tf.layers.max_pooling2d(conv2_2,
                                       (2, 2), # kernel size
                                       (2, 2), # stride
                                       name = 'pool2')

    conv3_1 = tf.layers.conv2d(pooling2,
                               32, # output channel number
                               (3,3), # kernel size
                               padding = 'same',
                               activation = activation,
                               kernel_initializer = kernel_initializer,
                               name = 'conv3_1')
    conv3_2 = tf.layers.conv2d(conv3_1,
                               32, # output channel number
                               (3,3), # kernel size
                               padding = 'same',
                               activation = activation,
                               kernel_initializer = kernel_initializer,
                               name = 'conv3_2')
    # 4 * 4 * 32
    pooling3 = tf.layers.max_pooling2d(conv3_2,
                                       (2, 2), # kernel size
                                       (2, 2), # stride
                                       name = 'pool3')
    # [None, 4 * 4 * 32]
    flatten = tf.layers.flatten(pooling3)
    return flatten
    

