
import pandas as pd
from tqdm import tqdm
import itertools
import numpy as np
import random
import copy
import time
import torch
tqdm.pandas(ascii=True)
import os
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append(os.path.abspath("./"))  # 添加 `models` 目录到 Python 路径
from termcolor import colored
from tangermeme.plot import plot_logo

from tangermeme.ism import saturation_mutagenesis
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append(os.path.abspath("./models/motifmodel"))  # 添加 `models` 目录到 Python 路径
from motifmodel_51 import Lucky as Lucky_51
from motifmodel_101 import Lucky as Lucky_101
from motifmodel_201 import Lucky as Lucky_201
from motifmodel_301 import Lucky as Lucky_301
from motifmodel_401 import Lucky as Lucky_401
from motifmodel_501 import Lucky as Lucky_501
from motifmodel_701 import Lucky as Lucky_701
from motifmodel_901 import Lucky as Lucky_901
from motifmodel_1001 import Lucky as Lucky_1001

sys.path.append(os.path.abspath("./"))
from plot import plot_logo
from Utils import generate_unique_filename

# 加载不同尺寸的模型
model_51 = Lucky_51().to(device)
model_101 = Lucky_101().to(device)
model_201 = Lucky_201().to(device)
model_301 = Lucky_301().to(device)
model_401 = Lucky_401().to(device)
model_501 = Lucky_501().to(device)
model_701 = Lucky_701().to(device)
model_901 = Lucky_901().to(device)
model_1001 = Lucky_1001().to(device)

# 定义映射选择类
model_classes = {
    51: Lucky_51,
    101: Lucky_101,
    201: Lucky_201,
    301: Lucky_301,
    401: Lucky_401,
    501: Lucky_501,
    701: Lucky_701,
    901: Lucky_901,
    1001: Lucky_1001,

}

params = {
    'batch_size': 128,
    'data_index': 10,  # 示例文件索引
    'seq_len': 501,    # 序列长度
    'seed': 2
}

# 处理碱基序列的办法, 输入字符串并且自动调整长度
def encode_dna_tensor(sequence):
    # 碱基映射规则
    mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    # 将 DNA 序列映射到索引
    encoded = [mapping[base] for base in sequence]
    # 转换为 PyTorch 张量，并调整形状为 (1, len(sequence))
    return torch.tensor(encoded).unsqueeze(0)

# 进行碱基字符串裁剪
def trim_seq(seq, length):
    if(length<51):
        return "" # 如果返回空序列, 代表目前的长度不太行
    elif(length>=51 and length<101):
        return seq[:51]
    elif(length>=101 and length<201):
        return seq[:101]
    elif(length>=201 and length<301):
        return seq[:301]
    elif(length>=301 and length<401):
        return seq[:301]
    elif(length>=401 and length<501):
        return seq[:401]
    elif(length>=501 and length<701):
        return seq[:501]
    elif(length>=701 and length<901):
        return seq[:701]
    elif(length>=901 and length<1001):
        return seq[:901]
    elif(length>=1001):
        return seq[:1001]


class SliceWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SliceWrapper, self).__init__()
        self.model = model
        # self.target = target

    def forward(self, X):
        # print(self.model(X))
        # print(X.shape)
        out = self.model(X)
        out = out[0][:, 1].unsqueeze(1)
        # print(out.shape)
        result = torch.sigmoid(out)
        #print(result)
        return result




#传入序列, 以及发生了何种修饰(以数组的形式保存), 对单一序列进行画图
def draw(seq,arr):
    # 准备输入
    seq=trim_seq(seq,len(seq)) # 还是得裁剪到规范的长度
    length=len(seq)


    # 设置颜色映射
    custom_colors = {
        'A': 'red',
        'C': 'blue',
        'G': 'orange',
        'U': 'green'
    }

    # 根据发生的第一种修饰来进行绘图
    if len(arr)>0 :
        index = arr[0]

        x = encode_dna_tensor(seq).to(device) # 转化为张量
        x = x.to(torch.int64)
        x = F.one_hot(x, num_classes=4).transpose(1, 2).float()
        model = model_classes[51]().to(device)  # 根据长度获取模型,默认使用长度51的模型
        model.load_state_dict(torch.load(f"./save/mymodel_{51}/data{index}.pth", map_location=device, weights_only=True)['state_dict'])  # 加载对应的模型
        wrapper = SliceWrapper(model)
        total_attr=None
        plt.rcParams.update({'font.size': 20})  # 设置全局字体大小

        # 默认设置长度为10
        plt.figure(figsize=(10, 5))

        # 这里得大修一下()
        if len(seq)==51:   # 这里做了一点点简单的修改
            for bit in range(length - 50):
                X_attr = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=0, end=50, device=device)
                if total_attr is None:
                    total_attr=X_attr
                else:
                    total_attr+=X_attr
                avg_attr = abs(total_attr)
                ax = plt.subplot(1, 1, 1)  # 5 行 1 列，选择第 1 个位置
                plot_logo(avg_attr[0, :, :], ax=ax, color=custom_colors)  # 绘制特征重要性 logo 图
                ax.set_title('Midpoint of the entire sequence:index=25', fontsize=14)
                ax.tick_params(axis='both', labelsize=12)  # 设置坐标轴字体大小
        else :
            num=(len(seq)-1)//100 #计算出一共有多少个图片
            for index in range(num):
                X_attr = saturation_mutagenesis(wrapper.cpu(), x.cpu(), start=index*100, end=(index+1)*100, device=device)
                total_attr = X_attr  # 初始化累加矩阵
                avg_attr = abs(total_attr)
                ax = plt.subplot(num, 1, index+1)  # 5 行 1 列，选择第 1 个位置
                plot_logo(avg_attr[0, :, :], ax=ax, color=custom_colors)  # 绘制特征重要性 logo 图
                ax.set_title(f'Midpoint of the sequence  :index={index*100+50}', fontsize=14)
                ax.tick_params(axis='both', labelsize=12)  # 设置坐标轴字体大小


        plt.tight_layout()
        # 保存图像在对应的文件夹里面
        filename=generate_unique_filename(extension=".svg")
        plt.savefig('./Files/'+filename)

        # 返回文件名称
        return filename


    else:
        plt.figure(figsize=(10, 6))  # 设定图片大小
        plt.gca().set_facecolor('white')
        plt.axis('off')
        plt.text(0.5, 0.5, "No relevant modifications detected",
                 fontsize=20, color='black', ha='center', va='center')
        plt.savefig('./motifs_combined.sv', format='svg')
        filename = generate_unique_filename(extension=".svg")
        plt.savefig('./Files/' + filename)

        # 返回文件名称
        return filename





