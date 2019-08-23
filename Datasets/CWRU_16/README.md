# CWRU_16

转速1730rpm、1750rpm、1772rpm、1797rpm，采样率12000samples/s。

原版官方数据，都是.mat文件，可以用python相关包导入。

数据集的理解可以参照这篇博客 https://blog.csdn.net/mibian9742/article/details/83047444 此处就不多废话。

数据集来源：美国凯斯西储大学实验室的轴承数据 http://csegroups.case.edu/bearingdatacenter/pages/12k-drive-end-bearing-fault-data

class             | label
:-----------------|------
0.007-Ball        | 0
0.007-InnerRace   | 1
0.007-OuterRace3  | 2
0.007-OuterRace6  | 3
0.007-OuterRace12 | 4
0.014-Ball        | 5
0.014-InnerRace   | 6
0.014-OuterRace6  | 7
0.021-Ball        | 8
0.021-InnerRace   | 9
0.021-OuterRace3  | 10
0.021-OuterRace6  | 11
0.021-OuterRace12 | 12
0.028-Ball        | 13
0.028-InnerRace   | 14
Normal            | 15