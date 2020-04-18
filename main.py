# -*- coding UTF-8 -*-
import tensorflow as tf
import os
from os import path
import numpy as np
import pickle
from data_feed import dataFeed
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
class resNet():
    def __init__(self):
        self.net_list = []
    def __dimCovn(self,x):
        input_chanel = x.get_shape().as_list()[-1]
        output_chanel = 2*input_chanel
        cov1 = tf.layers.conv2d(x,output_chanel,(3,3),
        strides=(2,2),activation=tf.nn.relu,padding='same',
        name='dim_cov1'
        )
        cov2 = tf.layers.conv2d(cov1,output_chanel,(3,3),
        strides=(1,1),activation=tf.nn.relu,padding='same',
        name='dim_cov2'
        )
        pool_x = tf.layers.average_pooling2d(x,(2,2),(2,2))
        pad_x = tf.pad(pool_x,[
            [0,0],
            [0,0],
            [0,0],
            [input_chanel//2,input_chanel//2],
        ])
        return cov2+pad_x   

    def __normalCovn(self,x):
        input_chanel = x.get_shape().as_list()[-1]
        cov1 = tf.layers.conv2d(x,input_chanel,(3,3),
        strides=(1,1),activation=tf.nn.relu,padding='same',
        name='normal_cov1'
        )
        cov2 = tf.layers.conv2d(cov1,input_chanel,(3,3),
        strides=(1,1),activation=tf.nn.relu,padding='same',
        name='normal_cov2'
        )
        return cov2+x

    def residual_block(self,x,output_chanel):
        input_chanel = x.get_shape().as_list()[-1]
        
        if input_chanel*2==output_chanel:
            covn =  self.__dimCovn(x)
        elif input_chanel==output_chanel:
            covn =  self.__normalCovn(x)
        else:
            raise Exception("output_chanel not match input chanel")
        return covn
    def __create_residual_layers(self,layer,layerDeep):
        x = self.net_list[-1]
        output_chanel = x.get_shape().as_list()[-1]*2
        for i in range(0,layerDeep):
            name = "residual_layer_{layer}_{deep}".format(layer=layer,deep=i)
            with tf.variable_scope(name):
                input_x = self.net_list[-1]
                residual_layer = self.residual_block(input_x,output_chanel)
                self.net_list.append(residual_layer)
        return self
    def creatNet(self,x_image,net_struct,base_chanel,class_num):
        with tf.variable_scope('cov0'):
            cov0 = tf.layers.conv2d(x_image,base_chanel,(1,1),strides=(1,1),
            padding='same',activation=tf.nn.relu,name="cov0")
            self.net_list.append(cov0)
        #构造残差结构
        for layer in range(0,len(net_struct)):
            self.__create_residual_layers(layer,net_struct[layer])
        residual_out_put = tf.reduce_mean(self.net_list[-1],[1,2])
        logits = tf.layers.dense(residual_out_put,class_num)
        self.net_list.append(logits)
        return self.net_list[-1]

                

            




#开始创建模型
x = tf.placeholder(tf.float32,[None,3072]) # (-1,3072)
y = tf.placeholder(tf.int64,[None]) #(-1,)
#(-1,3072)
x_image = tf.reshape(x,shape=(-1,3,32,32))
x_image = tf.transpose(x_image,perm=(0,2,3,1))

y_ = resNet().creatNet(x_image,[2,3,2],32,10)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)
#计算正确率
predict = tf.argmax(y_,1)
correct = tf.cast(tf.equal(predict,y),tf.float32)
accurancy = tf.reduce_mean(correct)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
init = tf.global_variables_initializer()
dataModel = dataFeed("data_batch_1").loadData()
with tf.Session() as sess:
    sess.run(init)
    for i in range(0,1000):
        datas,labels = dataModel.next_batch(128)
        v1,v2,_,v3,v4=sess.run(
            [loss,accurancy,train_op,predict,y],
            feed_dict={x:datas,y:labels}
        )
        
        
        msg="第{ep}轮训练：损失[{loss}],正确率[{accurancy}]".format(
                ep=i,
                loss=v1,
                accurancy=v2
        )
        print(msg)
            
