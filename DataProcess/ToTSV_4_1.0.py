#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Aug 30 15:17:00 2019
school:HUST
@author: KJ.Zhou
"""

#改进取样方法，随机选取样本的起始位置，然后从起始位置开始截取样本长度个采样点得到一个样本

from scipy.io import loadmat
import os
import random
import numpy as np
import pandas as pd

def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True

root = 'c:/Users/dreamby/Desktop/CWRU'
path = root+'/'+'Datasets/CWRU_4/1797'

filenames = os.listdir(path)
DE_times = []
#FE_times = []
Label_list = []
#由于本代码没有自动边界检测，所以下面样本长度以及样本数量以及从原文件中截取多少采样点自己设定的时候不要越界了。
sample_length = 400 #样本长度，根据需求修改
sample_number = 3000 #每个文件取样数量

for filename in filenames:

    label = filename[:-4]
    label = label[6:]#解析文件名，如果你要分成10类或者其他类别数，你需要修改此处代码以及下面打标签的代码，自己定义怎么分类。
    Label_list.append(label)
    filepath = path + '/' + filename
    m = loadmat(filepath)
    keys = list(m.keys())
    #print(keys) #可以查看.mat文件里面都有什么数据
    for key in keys:
        if 'DE_time' in key:#选择驱动端加速度计数据，如有需要可以得到其他加速度计数据
                index1 = key
        if 'FE_time' in key:
                index2 = key
    DE_time = m[index1]
    #FE_time = m[index2]
    if label!='Normal':
        DE_time = DE_time[0:120000]
        #FE_time = FE_time[0:120000]
    else:
        DE_time = DE_time[0:240000]   
        #FE_time = FE_time[0:240000]
    DE_times.append(DE_time)
    #FE_times.append(FE_time)
    
path_save = root+'/'+'Series/Single_CWRU_4/CWRU_1797'
mkdir(path_save)

for i in range(len(Label_list)):

        print(Label_list[i])
        if Label_list[i] == 'Normal':
                idx = 0
        elif Label_list[i] == 'Ball':
                idx = 1
        elif Label_list[i] == 'InnerRace':
                idx = 2
        else:
                idx = 3
        print(idx)

        records = []
        begins=random.sample(range(0,int(len(DE_times[i])*100000/120000)),int(len(DE_times[i])*sample_number/120000))
        
        for begin in begins:
                        sample1 = []
                        #sample2 = []
                        for m in range(sample_length):
                                sample1.extend(list(DE_times[i][begin+m]))
                                #sample2.extend(list(FE_times[i][begin+m]))
                        record = [idx] + sample1 #+ sample2
                        records.append(record) 

        temp = np.array(records)
        rate = 0.7#数据划分成训练集和测试集的比例
        length = int(rate*len(DE_times[i])*sample_number/120000)
        
        indices = np.random.permutation(temp.shape[0])
        train_idx, test_idx = indices[:length], indices[length:]#随机划分成训练集和测试集
        train, test = temp[train_idx,:], temp[test_idx,:]

        if i==0:
                trains = train
                tests = test
        else:
                trains = np.r_[trains,train]
                tests = np.r_[tests,test]
    
trains = pd.DataFrame(trains)
trains[0] = trains[0].astype(int) 
trains.to_csv(path_save+'/TRAIN.tsv',header=0,sep = '\t',columns=None,index=0)
print('trains save ')

tests = pd.DataFrame(tests)
tests[0] = tests[0].astype(int) 
tests.to_csv(path_save+'/TEST.tsv',header=0,sep = '\t',columns=None,index=0)
print('tests save ')
