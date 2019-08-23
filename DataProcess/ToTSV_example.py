#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Aug 30 15:17:00 2019
school:HUST
@author: KJ.Zhou
"""
#这是一个数据预处理的示例代码，你可以根据你的需求修改此代码，指定样本长度，指定样本数量，指定如何划分训练集和测试集，指定类别
#由于水平有限，代码难免写得有点冗余，可自行简化
from scipy.io import loadmat
import os
import numpy as np
import pandas as pd

def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True

#root为你整个项目的路径，其他文件可以参照我的目录结构
root = 'c:/Users/dreamby/Desktop/CWRU'
path = root+'/'+'Datasets/CWRU_4/1730'
#获取改目录下所有文件名，得到一个文件名列表
filenames = os.listdir(path)

#用于存放传感器数据的列表，根据需要修改，原始有三个传感器数据，但部分传感器数据不全
DE_times = []
#用于存放标签的列表，通过文件名解析获得
Label_list = []

sample_length = 400 #样本长度，根据需求修改
sample_number = 300 #每个文件取样数量

for filename in filenames:

    label = filename[:-4]
    label = label[6:]#解析文件名，目的是为了后面分类提供依据。如果你要分成10类或者其他类别数，你需要修改此处代码以及下面打标签的代码，自己定义怎么分类。
    Label_list.append(label)
    filepath = path + '/' + filename
    m = loadmat(filepath)
    keys = list(m.keys())
    #print(keys) #可以查看.mat文件里面都有什么数据
    for key in keys:
        if 'DE_time' in key:#选择驱动端加速度计数据，如有需要可以得到其他加速度计数据,也可以同时提取多个传感器数据
                index = key
    DE_time = m[index]
    if label!='Normal':
        DE_time = DE_time[0:sample_length*sample_number]
    else:
        DE_time = DE_time[0:sample_length*sample_number*2]   #正常数据采集了600个样本，每个样本长度为400
    DE_times.append(DE_time)
    
path_save = root+'/'+'Series/CWRU/CWRU_1730'
mkdir(path_save)

for i in range(len(DE_times)):
        print(Label_list[i])
        #自定义分类规则
        if Label_list[i] == 'Normal':
                idx = 0
        elif Label_list[i] == 'Ball':
                idx = 1
        elif Label_list[i] == 'InnerRace':
                idx = 2
        else:#OuterRace6
                idx = 3

        print(idx)

        records = []
        for j in range(int(len(DE_times[i])/sample_length)):
                        sample = []
                        for m in range(sample_length):
                                sample.extend(list(DE_times[i][j*sample_length+m]))
                        record = [idx] + sample
                        records.append(record)  
        temp = np.array(records)

        rate = 0.85#数据划分成训练集和测试集的比例
        train_number = int(rate*len(DE_times[i])/sample_length)#训练集的数量

        #随机划分成训练集和测试集
        indices = np.random.permutation(temp.shape[0])
        train_idx, test_idx = indices[: train_number], indices[train_number:]
        train, test = temp[train_idx,:], temp[test_idx,:]

        if i==0:
                trains = train
                tests = test
        else:
                trains = np.r_[trains,train]
                tests = np.r_[tests,test]

#保存处理好的数据
trains = pd.DataFrame(trains)
trains[0] = trains[0].astype(int) 
trains.to_csv(path_save+'/TRAIN.tsv',header=0,sep = '\t',columns=None,index=0)
print('trains save ')

tests = pd.DataFrame(tests)
tests[0] = tests[0].astype(int) 
tests.to_csv(path_save+'/TEST.tsv',header=0,sep = '\t',columns=None,index=0)
print('tests save ')
