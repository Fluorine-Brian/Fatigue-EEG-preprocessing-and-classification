# Fatigue-EEG-preprocessing-and-classification
Fatigue EEG dataset is collected by our devices. These codes are used to preprocess the dataset, generating a specific data structure for machine learning to work. Here, a few machine learning method is used and a SECNN is used for classification.

Environment:
conda env create -f freeze.yml

数据预处理步骤
1.按1秒的时间窗口截取样本 2.根据受试者数据采集阶段所填量表给样本打上对应标签 3.进行MNE滤波和伪迹去除 4.优化数据集质量，计算每个样本的theta波和beta波能量的比值，根据该比值筛选出优质样本组成最终的数据集 先python preprocess.py，然后python qualify.py

Dataset:
dataset/qualified_dataset.mat 总共2525个样本，每个样本均为时长1秒的前额三通道脑电数据，采样频率为250Hz，形状为3*250 清醒:中度疲劳:重度疲劳 = 1031:826:668 训练集:验证集:测试集 = 6:2:2

Training test:
通过args.py脚本调整训练参数和模型参数 python train.py

单个样本测试用例
demo.py 输入单个样本长度为1秒，采样频率250hz，三通道，形状为3*250。 输入数据需要先经过MNE滤波和伪迹去除预处理，之后reshape成(1,1,3,250)的形状 输出为预测的标签（疲劳状态）

Result:
Final testing result: 

Accuracy: 93.86%

Precision: 93.77%

Recall: 93.50%

F1score: 93.63%
