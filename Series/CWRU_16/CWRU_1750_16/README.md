# CWRU_1750_16

Train size: 4320

Test size: 480

Missing value: No

Number of classses: 16

Time series length: 400 

转速1750rpm，采样率12000samples/s，使用的是DE_time数据（驱动端加速度计数据）

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