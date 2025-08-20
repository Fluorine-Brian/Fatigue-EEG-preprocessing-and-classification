import torch
import numpy as np
from model import SECNN  # 确保模型结构已定义
import os

"""
输入单个样本长度为1秒，采样频率250hz，三通道，形状为3*250。
输入数据需要先reshape成(1,1,3,250)的形状
输出为预测的标签（疲劳状态）
"""


def classify_sample(sample, model_path='bestcheckpoint.pt', device='cuda'):
    # 加载模型结构
    my_net = SECNN().double().to(device)
    
    # 加载模型权重
    my_net.load_state_dict(torch.load(model_path))
    
    # 将模型切换到评估模式
    my_net.eval()
    
    # 处理输入样本：确保输入数据的形状与训练时一致
    sample = sample.reshape(1, 1, 3, 250)   # (1, 1, 3, 250)
    sample = torch.DoubleTensor(sample).to(device)
    
    # 进行预测
    with torch.no_grad():  # 禁用梯度计算，提高推理速度
        output = my_net(sample)
        predicted_label = output.argmax(dim=1).cpu().numpy()[0]  # 选择概率最大的一类
    
    return predicted_label


# 示例使用
if __name__ == '__main__':
    
    sample = np.random.randn(3, 250)  # 使用随机数据来模拟真实样本

    # 选择设备，cuda如果有GPU可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 使用加载的模型对样本进行分类
    pred = classify_sample(sample, model_path='bestcheckpoint.pt', device=device)
    
    labels_dict = {0: '清醒', 1: '中度疲劳', 2: '重度疲劳'}
    print(f"预测结果: {labels_dict[pred]} (标签: {pred})")
