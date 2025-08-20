from scipy.io import loadmat

features_struct = loadmat('C:/算法源代码及结果\疲劳范式实验数据/fatigue_dataset.mat')
print(features_struct['labels'])
