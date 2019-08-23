# CWRU_4

转速1730rpm、1750rpm、1772rpm、1797rpm，采样率12000samples/s。

自己从官方数据中挑选过的数据，都是.mat文件，可以用python相关包导入。你可以按照自己需求挑选数据

数据集的理解可以参照这篇博客 https://blog.csdn.net/mibian9742/article/details/83047444 此处就不多废话。

数据集来源：美国凯斯西储大学实验室的轴承数据 http://csegroups.case.edu/bearingdatacenter/pages/12k-drive-end-bearing-fault-data

class            | label
:----------------|------
0.007-Ball       | 1
0.007-InnerRace  | 2
0.007-OuterRace6 | 3
0.014-Ball       | 1
0.014-InnerRace  | 2
0.014-OuterRace6 | 3
0.000-Normal     | 0