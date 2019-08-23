#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Aug 30 15:17:00 2019
school:HUST
@author: KJ.Zhou
"""
#这是一个读取.tsv文件以及做相关处理的的示例代码，你可以根据你的需求修改此代码。
#此处是绘制了反应时间序列相关性的gaf图像（可用于后续卷积神经网络），至于什么是gaf图像，此处不做详解。
#由于水平有限，代码难免写得有点冗余，可自行简化
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField

def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True

# 数据集导入路径
path = 'C:/Users/dreamby/Desktop/CWRU/Series/CWRU_4'
dataset_name = 'CWRU_1797_4'
dataset_path = path + '/' + dataset_name
print(dataset_path)
test_file = dataset_path + '/' + 'TEST.tsv'
train_file = dataset_path + '/'  + 'TRAIN.tsv'

#导入数据
train_data = pd.read_csv(train_file, sep = '\t', header=None)
test_data = pd.read_csv(test_file, sep = '\t', header=None)
# Parameters
train_samples = train_data.shape[0] #训练集样本数量
train_timestamps = train_data.shape[1] - 1 #每个样本的采样点数量
test_samples = test_data.shape[0] #测试集样本数量
test_timestamps = test_data.shape[1] - 1 #每个样本的采样点数量

#dataset
X_train = []
Lables_train = []
X_test = []
Lables_test = []

# 数据导入到列表里面
for i in range(train_samples):
    times = train_data.iloc[i][1:]
    X_train.append(times)
    Lables_train.append(train_data.iloc[i][0])
for i in range(test_samples):
    times = test_data.iloc[i][1:]
    X_test.append(times)
    Lables_test.append(test_data.iloc[i][0])


# Transform the time series into Gramian Angular Fields
gasf = GramianAngularField(image_size=128, method='summation')
X_train_gasf = gasf.fit_transform(X_train)
X_test_gasf = gasf.fit_transform(X_test)

save_path = 'C:/Users/dreamby/Desktop/CWRU/Image' + '/' + dataset_name
mkdir(save_path + '/train')
mkdir(save_path + '/test')

image_name1 = []
image_name2 = []

# 生成图片并保存到相应文件夹下
for i in range(train_samples):
    
    plt.figure(figsize=(1.28, 1.28))
    image = X_train_gasf[i]
    title = str(i+1)+'.jpg'
    image_name1.append(title)
    plt.imshow(image, cmap='rainbow', origin='lower')
    plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.savefig(save_path + '/train' + '/' + title)
    plt.close()
    #plt.show()

# 生成相应的标签并保存为.csv
data1 = {'imageName':image_name1, 'label':Lables_train}
dt1 = pd.DataFrame(data1)
dt1.to_csv(save_path + '/train_label.csv')


# 生成图片并保存到相应文件夹下
for i in range(test_samples):
    
    plt.figure(figsize=(1.28, 1.28))
    image = X_test_gasf[i]
    title = str(i+1)+'.jpg'
    image_name2.append(title)
    plt.imshow(image, cmap='rainbow', origin='lower')
    plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.savefig(save_path + '/test' + '/' + title)
    plt.close()
    #plt.show()
    
# 生成相应的标签并保存为.csv
data2 = {'imageName':image_name2, 'label':Lables_test}
dt2 = pd.DataFrame(data2)
dt2.to_csv(save_path + '/test_label.csv')