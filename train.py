#-*- coding UTF-8-*-
import os
import tensorflow as tf
from networks.vggnet import mk_vgg_net
from models.train_model import trainOS

class train(trainOS):
    def __init__(self,model_name):
        dRoot = os.path.abspath(os.path.dirname(__file__))
        super().__init__(model_name,dRoot)
        self.LEARNING_RATE = 1e-3        #学习率
        self.TEST_STEP = 10              #测试点
        self.CKPT_STEP = 100             #保存点
        self.SUMMARY_STEMP = 10          #记录点
        self.ECOPE = 1000                #训练次数
        #初始化
        self.__compile_net()
        self.load_model()
        self.__calculate()
    def __compile_net(self):
        #创建x_image
        self.x_inputs = tf.placeholder(tf.float32,(None,32*32*3))
        self.label = tf.placeholder(tf.int64)
        self.is_training = tf.placeholder(tf.bool)
        tempInputs = tf.reshape(self.x_inputs,(-1,3,32,32))
        self.x_image = tf.transpose(tempInputs,(0,2,3,1))
        self.x_image_input = self.x_image/127.5-1
        self.activation = tf.nn.relu
        self.kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
        self.cnn_net = mk_vgg_net(self.x_image_input,self.is_training,self.activation,self.kernel_initializer)
        self.flatten = self.cnn_net.createNet()
        return self
    def __calculate(self):
        #计算预测值
        self._y = tf.layers.dense(self.flatten,10)
        self.predict = tf.arg_max(self._y,1)
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.label,logits=self._y)
        self.summary_loss = tf.summary.scalar("loss_summary",self.loss)
        #计算正确率
        accurancy = tf.equal(self.label,self.predict)
        accurancy = tf.cast(accurancy,tf.float32)
        self.accurancy = tf.reduce_mean(accurancy)
        self.summary_accurancy = tf.summary.scalar("accurancy",self.accurancy)
        self.cnn_net.summary()
        #定义梯度下降算法
        with tf.name_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.SESSION.run(init)
        return self
    def train_default(self,step):
        if step%self.SUMMARY_STEMP==0:
            summaries = tf.summary.merge_all()
            options = [self.loss,self.accurancy,self.train_op,summaries]
        else:
            options = [self.loss,self.accurancy,self.train_op]
        data,label = self.next_batch()
        result=self.SESSION.run(options,feed_dict={
            self.x_inputs:data,
            self.label :label,
            self.is_training:True
        })
        loss_va,accurancy_va = result[0:2]
        msg = "训练%d:loss=%.2f,accurancy=%.2f"%(step,loss_va,accurancy_va)
        end = "\n" if step%self.TEST_STEP==0 else "\r"
        if step%self.SUMMARY_STEMP==0:
            summary_str = result[-1]
            train_writer,_ = self.getWriter()
            train_writer.add_summary(summary_str,global_step=step)
        print(msg,end=end)
    def train_test(self,step):
        if step%self.TEST_STEP:
            return None
        if step%self.SUMMARY_STEMP==0:
            summaries = tf.summary.merge([self.summary_loss,self.summary_accurancy])
            options = [self.loss,self.accurancy,summaries]
        else:
            options = [self.loss,self.accurancy]
        data,label = self.fix_batch()
        result=self.SESSION.run(options,feed_dict={
            self.x_inputs:data,
            self.label :label,
            self.is_training:False
        })
        loss_va,accurancy_va = result[0:2]
        msg = "测试%d:loss=%.2f,accurancy=%.2f"%(step,loss_va,accurancy_va)
        if step%self.SUMMARY_STEMP==0:
            summary_str = result[-1]
            _,test_writer = self.getWriter()
            test_writer.add_summary(summary_str,global_step=step)
        print(msg)
    def run(self):
        for i in range(self.ECOPE):
            step = i+1
            self.train_default(step)
            self.train_test(step)
            if step%self.CKPT_STEP==0:
                self.save_model(step)
        else:
            self.SESSION.close()

if __name__ == "__main__":
    tModel = train("VGG_4")
    tModel.run()



