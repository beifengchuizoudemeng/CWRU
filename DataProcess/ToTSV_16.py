#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Aug 30 15:17:00 2019
school:HUST
@author: KJ.Zhou
"""
from scipy.io import loadmat
import os
import numpy as np
import pandas as pd

#归一化到[-1,1]
def Normalize(data):
    max = np.max(data)
    min = np.min(data)
    result = (2 * data - min - max) / (max - min)
    result = np.where(result >= 1., 1., result)
    result = np.where(result <= -1., -1., result)
    return result


def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True

root = 'c:/Users/dreamby/Desktop/CWRU'#改成自己电脑下的相应的路径
path = root+'/'+'Datasets/CWRU_16/1730'

filenames = os.listdir(path)
DE_times = []
Label_list = []

for filename in filenames:

    label = filename[:-4]
    Label_list.append(label)
    filepath = path + '/' + filename
    m = loadmat(filepath)
    keys = list(m.keys())
    #print(keys) #可以查看.mat文件里面都有什么数据
    for key in keys:
        if 'DE_time' in key:#选择驱动端加速度计数据，如有需要可以得到其他加速度计数据
                index = key
    
    DE_time = m[index]
    DE_time = DE_time[0:120000]#因为每种类型只生成300个样本，每个样本长度为400.所以此处是120000，根据自己需求修改
    #DE_time = Normalize(DE_time) #归一化根据自己需求弄，不过此处的归一化只是把相同类型的归一化了，没有一起归一化。前者的归一化会丢失部分振幅信息（因为不同类型，振幅差异比较大），但利于模型迁移。
    # 后者的统一归一化，会对单一工况下的性能有提升。不过其实前者在单一工况下也是可以做到基本百分百分类准确率的。更多信息有待挖掘。
    DE_times.append(DE_time)
    
path_save = root+'/'+'Series/CWRU_16/CWRU_1730_16'#生成的文件保存路径
mkdir(path_save)

for i in range(len(DE_times)):#len(DE_times)

        print(Label_list[i])
        print(i)
        records = []
        for j in range(300):
                sample = []
                for m in range(400):
                        sample.extend(list(DE_times[i][j*400+m]))
                record = [i] + sample
                records.append(record)  

        temp = np.array(records)
        #rate = 0.9 #270是300*0.9，训练集：测试集9：1，可以根据需求改
        indices = np.random.permutation(temp.shape[0])
        train_idx, test_idx = indices[:270], indices[270:]#随机划分成训练集和测试集
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
