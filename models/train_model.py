#-*- coding UTF-8-*-
import os
import tensorflow as tf
from .cifar10_data import *
class trainOS():
    def __init__(self,model_name,dRoot):
        
        self.SESSION = tf.Session()
        self.MODEL_NAME = model_name
        self.D_ROOT = dRoot
        #初始化文件夹
        self.__init_dirs()
        #初始化数据
        self.BATCH_SIZE = 20
        self.FIX_BATCH = test_data.next_batch(self.BATCH_SIZE)
        #tensorFlow 对象
        self.train_writer = False
        self.test_writer = False
        
    def __get_dirs(self,paths):
        res_path = self.D_ROOT
        res_path = os.path.join(self.D_ROOT,paths)
        if not os.path.isdir(res_path):
            os.makedirs(res_path)
        return res_path
    def __clearn_dir(self,path):
        for f in os.listdir(path):
            os.remove(os.path.join(path,f))
    def __init_dirs(self):#初始化文件夹
        self.P_RUNTIME = self.__get_dirs("runTime")
        self.P_MODEL = self.__get_dirs(os.path.join(self.P_RUNTIME,self.MODEL_NAME))
        #建立目录
        self.P_TRAIN = self.__get_dirs(os.path.join(self.P_MODEL,"train"))
        self.P_TEST = self.__get_dirs(os.path.join(self.P_MODEL,"test"))
        #删除文件
        self.__clearn_dir(self.P_TEST)
        self.__clearn_dir(self.P_TRAIN)
        self.P_CKPT = self.__get_dirs(os.path.join(self.P_MODEL,"ckpt"))
        self.Saver = False
    def next_batch(self):
        return train_data.next_batch(self.BATCH_SIZE)
    def fix_batch(self):
        return self.FIX_BATCH
    def save_model(self,step):
        self.Saver =self.Saver if self.Saver else tf.train.Saver(max_to_keep=3)
        fpath = os.path.join(self.P_CKPT,"ckpt")
        self.Saver.save(self.SESSION,fpath,step)

    def load_model(self):
        self.Saver =self.Saver if self.Saver else tf.train.Saver(max_to_keep=3)
        ckpt = tf.train.get_checkpoint_state(self.P_CKPT)
        if ckpt and ckpt.model_checkpoint_path:
            res=self.Saver.restore(self.SESSION,ckpt.model_checkpoint_path)
            print("加载训练数据%s"%(ckpt.model_checkpoint_path))

    def getWriter(self):
        
        if not self.train_writer:
            self.train_writer = tf.summary.FileWriter(self.P_TRAIN,self.SESSION.graph)
            self.test_writer = tf.summary.FileWriter(self.P_TEST)
        return self.train_writer,self.test_writer
    def test(self):
        self.__init_dirs()
    
    
        
    
    
    
    
    
        

