import scipy.io as sio
import numpy as np
import os


def save_random_samples(data_dir='/home/lab-wu.yanzhang/workfile/fatigue-detection/dataset.mat', num_samples=10, save_dir='./samples'):
    # 创建保存样本的目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据集
    tmp = sio.loadmat(data_dir)
    xdata = np.array(tmp['samples'])
    ydata = np.array(tmp['labels']).flatten()

    # 随机选择样本的索引
    indices = np.random.choice(len(xdata), num_samples, replace=False)

    # 保存每个样本为单独的.mat文件
    for i, idx in enumerate(indices):
        sample = xdata[idx]  # 获取样本
        label = ydata[idx]  # 获取标签

        # 保存为 .mat 文件
        sample_data = {'sample': sample, 'label': label}
        sample_filename = os.path.join(save_dir, f'sample_{i+1}.mat')
        sio.savemat(sample_filename, sample_data)

        print(f'Saved sample {i+1} to {sample_filename}')


# 示例调用：从数据集中随机采样10个样本并保存
save_random_samples()
