# -*- coding UTF-8 -*-
import os
import tensorflow as tf
from vggnet import convnet
from cifar10_data import *

x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])
# [None], eg: [0,5,6,3]
x_image = tf.reshape(x, [-1, 3, 32, 32])
# 32*32
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])
#初始化网络定义损失函数
flatten = convnet(x_image,tf.nn.relu,tf.truncated_normal_initializer(stddev=0.02))
y_ = tf.layers.dense(flatten, 10)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
loss_summary = tf.summary.scalar('loss',loss)

predict = tf.argmax(y_, 1)
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
accuracy_summary = tf.summary.scalar('accuracy',accuracy)
source_image = (x_image + 1) * 127.5
inputs_summary = tf.summary.image('inputs_image', source_image)
merged_summary = tf.summary.merge_all()
merged_summary_test = tf.summary.merge([loss_summary, accuracy_summary])
#使用adam优化器进行训练
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

#开始训练/简历tensorbord记录
run_time_path = os.path.join(dRoot,"runTime")
train_path = os.path.join(run_time_path,'train')
test_path = os.path.join(run_time_path,'test')
if not os.path.isdir(train_path):
    os.makedirs(train_path)
if not os.path.isdir(test_path):
    os.makedirs(test_path)

#初始化参数
init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 100
output_summary_every_steps = 100
train_filenames = ""

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(train_path,sess.graph)
    test_writer =  tf.summary.FileWriter(test_path)
    fixed_test_batch_data, fixed_test_batch_labels = test_data.next_batch(batch_size)
    for i in range(train_steps):
        batch_data,batch_labels = train_data.next_batch(batch_size)
        #需要执行的基本命令
        eval_ops = [loss, accuracy, train_op] 
        should_output_summary = ((i+1) % output_summary_every_steps == 0)
        if should_output_summary:
            eval_ops.append(merged_summary)
        eval_ops_results=sess.run(eval_ops,feed_dict={
            x: batch_data,
            y: batch_labels
        })
        loss_va,accuracy_va = eval_ops_results[0:2]
        train_msg = "训练{stemp},loss={loss},accuracy={accuracy}".format(
            stemp=i+1,
            loss=loss_va,
            accuracy=accuracy_va)
        if should_output_summary:
            
            train_summary_str = eval_ops_results[-1]
            train_writer.add_summary(train_summary_str,i+1)
            test_summary_str,loss_va,accuracy_va= sess.run([merged_summary_test,loss,accuracy],
                                        feed_dict={
                                            x: fixed_test_batch_data,
                                            y: fixed_test_batch_labels,
                                        })
            test_writer.add_summary(test_summary_str,i+1)
            test_message = "测试{stemp},loss={loss},accuracy={accuracy}".format(
            stemp=i+1,
            loss=loss_va,
            accuracy=accuracy_va)
            print(train_msg)
            print(test_message)
            


        
        










