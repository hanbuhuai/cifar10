import os,pickle
from os import path
import numpy as np
dRoot = path.abspath(path.dirname(__file__))

class dataFeed():
    def __init__(self,fname,need_shuffle=True):
        self.__need_shuffle = need_shuffle #是否可以打乱
        self.__fname = fname    #文件明
        self.__pData = path.join(dRoot,"cifar-10-batches-py") #cifar10文件夹
        self.__datas = None #数据
        self.__labels= None #标签
        self.__indicator = 0 #循环计数器

    def __loadFile(self):
        fPath = path.join(self.__pData,self.__fname)
        with open(fPath,'rb') as fp:
            cPickle = pickle._Unpickler(fp)
            cPickle.encoding = "latin1"
            data = cPickle.load()
        return data['data'],data['labels']
    def __shuffle(self):
        if not self.__need_shuffle:#不允许打乱直接返回
            return False
        p = np.random.permutation(self.__datas.shape[0])
        self.__datas = self.__datas[p]
        self.__labels = self.__labels[p]
        self.__indicator = 0
        return True
    def next_batch(self,batch_size,end_shuffle=False):
        start_indecator = self.__indicator
        end_indecator=self.__indicator+batch_size
        #print(start_indecator,end_indecator)
        if end_indecator>self.__datas.shape[0]:
            if self.__shuffle() and not end_shuffle:
                return self.next_batch(batch_size,end_shuffle=True)
            else:
                raise Exception("no more data set")
        datas = self.__datas[start_indecator:end_indecator]
        labels= self.__labels[start_indecator:end_indecator]
        self.__indicator = end_indecator
        return datas,labels
    def loadData(self):
        data_all,label_all = list(),list()
        datas,labels = self.__loadFile()
        for data,label in zip(datas,labels):
            if label in (0,1) or True:
                data_all.append(data)
                label_all.append(label)
        self.__datas = np.vstack(data_all)/127.5-1
        self.__labels= np.hstack(label_all)
        return self

if __name__ == "__main__":
    # dataM = dataFeed('data_batch_1',True).loadData()
    # for i in range(0,1):
    #     datas,labels = dataM.next_batch(128)
    #     print(datas.shape,labels.shape)
    
    # a = 1
    # print(a,id(a))
    # def add_1(a):
    #     a= a+1
    #     print(a,id(a))
    # add_1(a)
    # print(a,id(a))

    # b = [1,2,3,4]
    # print(b,id(b))
    # def add_2(b):
    #     b+=b
    #     print(b,id(b))
    # add_2(b)
    # print(b,id(b))
    
    a=1
    b=[1,2]
    print(type(a))
    print(type(b))


    



