# -*- coding UTF-8 -*-
import tensorflow as tf
'''
   vggnet 网络搭建
'''
class mk_vgg_net():
    def __init__(self,inputs,is_training,activation,kernel_initializer):
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.inputs = inputs
        self.is_training = is_training
        self.net = list()
    def __conv(self,inputs,idx):
        name = "covn_%d_%d"%idx
        covn = tf.layers.conv2d(
            inputs,32,(3,3),
            padding='same',
            #activation=self.activation,
            kernel_initializer = self.kernel_initializer,
            name = name
        )
        bn = tf.layers.batch_normalization(covn,training=self.is_training)
        bn = self.activation(bn)
        self.net.append({name:bn})
        
        return name,covn
    def __pooling(self,inputs,idx):
        name = "pooling_%d_%d"%idx
        pooling = tf.layers.max_pooling2d(inputs,(2,2),(2,2),name=name)
        self.net.append({name:pooling})
        
        return name,pooling
    def vggWarper(self,deep=3):
        output = self.inputs
        for i in range(deep):
            name,output = self.__conv(output,(i,1))
            name,output = self.__conv(output,(i,2))
            name,output = self.__pooling(output,(i,1))
        return output
    def createNet(self,deep=3):
        output = self.vggWarper(deep)
        flatten = tf.layers.flatten(output)
        return flatten
    def summary(self):
        with tf.name_scope('vggnet_summary'):
            for item in self.net:
                name = list(item.keys())[0]
                layer= item[name]
                mean = tf.reduce_mean(layer)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(mean-layer)))
                tf.summary.scalar(name+'_mean', mean)
                tf.summary.scalar(name+'_stddev', stddev)
                tf.summary.scalar(name+'_min', tf.reduce_min(layer))
                tf.summary.scalar(name+'_max', tf.reduce_max(layer))
                tf.summary.histogram(name+'_histogram', layer)

