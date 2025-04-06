import torch
import torch.nn.functional as F
import sys
import os
import random
import itertools
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from torch.utils.data import DataLoader, TensorDataset
from termcolor import colored
import json
import cairosvg

sys.path.append(os.path.abspath("./models/models"))  # 添加 `models` 目录到 Python 路径
from mymodel import Lucky
from mymodel_51 import Lucky as Lucky_51
from mymodel_101 import Lucky as Lucky_101
from mymodel_201 import Lucky as Lucky_201
from mymodel_301 import Lucky as Lucky_301
from mymodel_401 import Lucky as Lucky_401
from mymodel_501 import Lucky as Lucky_501
from mymodel_701 import Lucky as Lucky_701
from mymodel_901 import Lucky as Lucky_901
from mymodel_1001 import Lucky as Lucky_1001

sys.path.append(os.path.abspath("./"))  # 添加 `models` 目录到 Python 路径
from motif import draw
# 转化为base64的方法
import base64
from PIL import Image
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 在模型定义时, 这是长度为501的模型....
model = Lucky().to(device)

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


# 在使用这个方法之前, 要先判断csv中的数据是否正常, 如果不正常要进行返回
# 读取csv的方法如下, 只读取seq的内容, 并且返回序列的长度
def read_file(file):
    datas = pd.read_csv(f"./1-1.csv")
    seq = list(datas['data'])

    seq = [s.replace(' ', '').replace('U', 'T') for s in seq]
    # seq的结构为: 159条数据, 每一条都为1001长度
    print("输入的序列个数为:",len(seq), "序列长度为:", len(seq[0]))
    seq = [encode_dna_tensor(x) for x in seq]
    print("处理后的数组:", seq[0])
    return seq,len(seq)


def image_to_base64(image_path):
    if image_path.lower().endswith(".svg"):
        # 把 SVG 转换为 PNG
        png_image = BytesIO()
        cairosvg.svg2png(url=image_path, write_to=png_image)
        png_image.seek(0)  # 复位流位置
        img = Image.open(png_image)
    else:
        img = Image.open(image_path)

    # 将图片转换为 Base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class Container_message:
    pass


# 处理碱基序列的方法
def encode_dna_tensor(sequence):
    # 碱基映射规则
    mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    # 将 DNA 序列映射到索引
    encoded = [mapping[base] for base in sequence]
    # 转换为 PyTorch 张量，并调整形状为 (1, len(sequence))
    return torch.tensor(encoded).unsqueeze(0)


# out, atts, vq_loss, x_recon, perplexity, x_final = model(encode_dna_tensor(sequence))

# 打印输出信息
# print("out shape:", out)  # 预测结果
# print("atts length:", len(atts))  # 注意力信息
# print("vq_loss:", vq_loss.item())
# print("x_recon shape:", x_recon.shape)  # 可能是重建数据
# print("perplexity:", perplexity.item)
# print("x_final shape:", x_final)  # 经过 Transformer 层后的特征


# 截取对应长度的数据?
def trim_seq(seq, length):
    if (length < 51):
        return ""  # 如果返回空序列, 代表目前的长度不太行
    elif (length >= 51 and length < 101):
        return seq[:51]
    elif (length >= 101 and length < 201):
        return seq[:101]
    elif (length >= 201 and length < 301):
        return seq[:301]
    elif (length >= 301 and length < 401):
        return seq[:301]
    elif (length >= 401 and length < 501):
        return seq[:401]
    elif (length >= 501 and length < 701):
        return seq[:501]
    elif (length >= 701 and length < 901):
        return seq[:701]
    elif (length >= 901 and length < 1001):
        return seq[:901]
    elif (length >= 1001):
        return seq[:1001]


# 根据长度选择对应的模型以后, 将seq和数据张量一同传入
def predict(model, seq_arr, length):


    container = Container_message()  # 生成相关的对象

    meth = [0,0,0,0,0,0,0,0,0,0]
    index_meth = []

    # 进行多重预测
    for seq in seq_arr:
        # 对每一条进行十种预测
        for index in range(1, 11): # 十个模型分别进行预测
            print("加载的模型为", f"./save/mymodel_{length}/data{index}.pth")
            model.load_state_dict(torch.load(f"./save/mymodel_{length}/data{index}.pth", map_location=device, weights_only=True)['state_dict'])  # 加载对应的模型
            model.eval()  # 设为评估模式
            model.to(device)
            seq = seq.to(device)  # 确保输入数据在正确的设备
            out, atts, vq_loss, x_recon, perplexity = model(seq)

            if out[0, 0].item() < out[0, 1].item(): # 发生了这种甲基化
                meth[index - 1] = 1
                if index not in index_meth:
                    index_meth.append(index) # 如果之前没发生过这种甲基化就要进行一下记录了



    container.meth = meth
    container.index_meth = index_meth

    # 甲基化数组,如果是0则
    return container



from websocket_server import WebsocketServer


def new_client(client, server):
    print("CSV版本为您服务:", client['id'])


# 当接收到消息时调用的回调函数
def message_received(client, server, message):
    print("接收到消息:", message)
    try:
        # 解析 JSON 数据
        data = json.loads(message)
        print("解析后的数据:", data)
        # 获取数据, 并且发送信息什么的
        seq = data.get("seq")
        print("修剪前长度", len(seq))
        seq = trim_seq(seq, len(seq))  # 裁剪到合适的长度
        print("修剪后长度", len(seq))
        print(seq)

        if seq == "":
            return jsonify({"message": f"incorrect length of sequence"})

        model = model_classes[len(seq)]().to(device)  # 根据长度获取模型
        container = predict(model, seq)  # 获取发生了何种甲基化, 以及对应甲基化的索引
        draw(seq, container.index_meth)  # 获取序列以及索引实现画图
        # 画完的图像会保存在 motif_combined.svg 里面
        base = image_to_base64("motifs_combined.svg")

        response_data = {
            "message": "OK",
            "meth": json.dumps(container.meth),  # 将容器的方法转换为 JSON 字符串
            "image": base
        }

        # 将字典转换为 JSON 字符串
        response_json = json.dumps(response_data)

        # 发送 JSON 响应
        server.send_message(client, response_json)
    except ConnectionResetError:
        print(f"连接被重置，客户端 ID: {client['id']}")
    except json.JSONDecodeError:
        print("无效数据!")

# 服务器啥的暂时不能启动了
# 创建 WebSocket 服务器
#server = WebsocketServer(host='0.0.0.0', port=8081)
# 注册回调函数
#server.set_fn_new_client(new_client)
#server.set_fn_message_received(message_received)
# 启动服务器
#print("WebSocket 服务器正在运行，访问 ws://localhost:8081")
#server.run_forever()

seq, length=read_file("测试");

# 开发到一半, 先不管了