# CWRU_1797_4

Train size: 16800

Test size: 7200

Missing value: No

Number of classses: 4

Time series length: 400

转速1797rpm，采样率12000samples/s，使用的是DE_time数据（驱动端加速度计数据）

样本划分说明，选取一个文件中前120000个采样点，随机从前100000里面挑选3000个起始点，从这些起始点出发取连续400个点作为一个样本。Normal选取的是文件中的前240000个采样点，随机从前200000里面挑选了6000个起始点。

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