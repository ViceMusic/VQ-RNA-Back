import torch
import torch.nn.functional as F
import sys
import os
import random
import itertools
import numpy as np
from flask import Flask, request, jsonify
import json
import cairosvg
from websocket_server import WebsocketServer
import base64
from PIL import Image
from io import BytesIO

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

sys.path.append(os.path.abspath("./"))  # 加载绘制motif的文件
from motif import draw
from FileControl import FileControl
from Utils import *
from Pool import SimpleThreadPool
from DataBase import Database

# 设置设备类型, 有GPU就用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载不同尺寸的模型,虽然除了51以外基本没啥用了, 不过还是加载上吧
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

# 生成文件调用对象
fileControl = FileControl()
# 生成线程池对象
pool = SimpleThreadPool(num_threads=5)
# 生成数据库操作对象
db = Database()


# 文件返回对象
class Container_message:
    pass


# 输入图片路径, 转化为base64格式
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


# 直接将一整个字符串转化为张量格式
def encode_dna_tensor(sequence):
    # 碱基映射规则
    mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    # 将 DNA 序列映射到索引
    encoded = [mapping[base] for base in sequence]
    # 转换为 PyTorch 张量，并调整形状为 (1, len(sequence))
    return torch.tensor(encoded).unsqueeze(0)


# 面向审稿人编程(划掉), 用来检测是否发生了某些不对的情况
def pan(site, type_des):
    # 定义修饰与碱基的对应关系
    mod_to_base = {
        1: 'A',  # Am (2'-O-甲基腺苷)
        2: 'C',  # Cm (2'-O-甲基胞苷)
        3: 'G',  # Gm (2'-O-甲基鸟苷)
        4: 'U',  # Um (2'-O-甲基尿苷)
        5: 'A',  # m1A (1-甲基腺苷)
        6: 'C',  # m5C (5-甲基胞苷)
        7: 'U',  # m5U (5-甲基尿苷)
        8: 'A',  # m6A (6-甲基腺苷)
        9: 'A',  # m6Am (N6,2'-O-二甲基腺苷)
        10: 'U'  # Ψ (假尿苷)
    }

    # 检查 type_des 是否在有效范围内
    if type_des < 1 or type_des > 10:
        return False

    # 获取修饰对应的碱基
    expected_base = mod_to_base[type_des]

    # 检查传入的碱基是否与修饰对应
    if site.upper() != expected_base:
        return False

    # 如果都匹配，返回 True
    return True


# 对于一个序列进行预测的方法
def predict_window(model, seq):
    print("进行序列预测, 序列长度:", len(seq))
    container = Container_message()  # 生成相关的对象
    char_seq = seq
    length = len(seq)
    seq = encode_dna_tensor(seq)  # 准备模型的输入
    meth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 判断发生了何种甲基化
    index_meth = []  # 发生甲基化的类型
    detail = []  # 三元组[发生甲基化位点, 碱基类型, 发生的修饰类型]

    # 这次只需要加载模型即可, 不需要进行模型选择了

    # 进行多重预测
    for index in range(1, 11):
        model.load_state_dict(
            torch.load(f"./save/mymodel_{51}/data{index}.pth", map_location=device, weights_only=True)[
                'state_dict'])  # 加载对应的模型
        model.eval()  # 设为评估模式
        model.to(device)
        seq = seq.to(device)  # 确保输入数据在正确的设备

        # 对第index种类修饰进行预测, 遍历整个序列
        for bit in range(length - 50):
            out, atts, vq_loss, x_recon, perplexity = model(seq[:, bit:bit + 50])
            if out[0, 0].item() < out[0, 1].item():  # 发生了这种甲基化
                if pan(char_seq[bit + 25], index):  # 还要判断一下这种甲基化是否合理
                    detail.append([bit + 25, index, char_seq[bit + 25]])
                    if index not in index_meth:
                        index_meth.append(index)  # 如果之前没发生过这种甲基化就要进行一下记录了
                        meth[index - 1] = 1

    # 封装具体的信息
    container.meth = meth
    container.index_meth = index_meth
    container.detail = detail

    return container


def new_client(client):
    print("新的客户端连接, 暂时不支持多线程运行:", client['id'])


def expose_sentence(seq, message):
    # 获取用户信息
    data = json.loads(message)
    email = data.get("email")

    print("为什么", len(seq))

    # 检查, 如果长度不够或者内容没有, 就返回错误信息
    if len(seq) < 51:
        response = json.dumps(
            {"type": "notice", "message": "error", "content": "incorrect length of sequence, please > 51 characters"})
        #server.send_message(client, response)
        return response

    _, userid = db.user_existed(email)  # 查询得到用户id
    _, taskid, _ = db.add_task(userid, " ", "in_progress")  # 获得任务信息

    # 否则, 如果序列长度>51, 则自动加载51这个模型.
    model = model_classes[51]().to(device)
    container = predict_window(model, seq)  # 进行预测
    svgName = draw(seq, container.index_meth)  # 进行画图,并且得到对应的文件名称

    print("检查", svgName)

    # 画完的图像会保存在对应的svg里面, 转化完成以后, 就删除这个文件, 就当临时出现了一次
    base = image_to_base64("./Files/" + svgName)
    fileControl.delete_file(svgName)  # 在转化为base64以后, 再次删除内容

    # 准备返回体内容
    response_data = {
        "message": "OK",
        "type": "single",  # 代表单一的序列, 代表无论detail也好, meth也好, 都是针对单一seq的
        "seqs": seq,  # 返回原本的信号类型, 如果是文件就需要别的东西了
        "meth": json.dumps(container.meth),  # 将容器的方法转换为 JSON 字符串
        "image": base,
        "detail": json.dumps(container.detail)
    }
    response_json = json.dumps(response_data)  # json转化为字符串格式, 方便websocket进行传输
    db.update_task(taskid, "completed", response_json)  # 任务执行结束, 输入库中
    #server.send_message(client, response_json)
    return response_json


def expose_fasta(fasta_name, message):
    # 获取用户信息
    data = json.loads(message)
    email = data.get("email")

    fasta_path = fileControl.find_file(fasta_name)
    # 对fasta进行处理的方法不太一样
    seqList = fasta_to_seq_list(fasta_path)  # 先根据fasta进行获取, 得到seq序列
    print("读取到的序列数目为", len(seqList))
    # 读取完fasta以后就删除文件
    fileControl.delete_file(fasta_name)

    # 检查, 如果长度不够或者内容没有, 就返回错误信息
    if len(seqList) == 0 or len(seqList[0]) < 51:
        response = json.dumps(
            {"type": "notice", "message": f"error", "content": "incorrect length of sequence, please > 51 characters"})
        #server.send_message(client, response)
        return response

    # 准备内容
    meths = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    details = []
    meth_index = []

    print(email)
    _, userid = db.user_existed(email)  # 查询得到用户id
    print(userid)
    _, taskid, _ = db.add_task(userid, " ", "in_progress")  # 获得任务信息

    # seq为 "AGCTAGCT", index为 0
    for seq in seqList:
        print("预测中")
        model = model_classes[51]().to(device)
        container = predict_window(model, seq)  # 进行预测, mesh查看发生了何种甲基化, detail是第几行发生了什么修饰
        # 更新甲基化种类
        print(container.meth)
        for i in range(len(container.meth)):
            if container.meth[i] == 1 and meths[i] == 0:
                meths[i] = 1
        # 增加具体细节
        details.append(container.detail)  # 这样就不需要index了

    # d
    for index in range(len(meths)):
        if meths[index] == 1:
            meth_index.append(index + 1)

    print("绘图中")
    # 绘图, 并且获取文件名称
    filename = draw(seqList[0], meth_index)
    # 获取为base64
    base = image_to_base64("./Files/" + filename)
    # 在转化为base64以后, 再次删除内容
    fileControl.delete_file(filename)

    # 准备返回体内容
    response_data = {
        "message": "OK",
        "type": "mutil",  # 代表单一的序列, 代表无论detail也好, meth也好, 都是针对单一seq的
        "seqs": seqList,  # 返回原本的信号类型, 如果是文件就需要别的东西了
        "meth": meths,  # 将容器的方法转换为 JSON 字符串
        "image": base,
        "detail": details
    }

    print(response_data)
    response_json = json.dumps(response_data)  # json转化为字符串格式, 方便websocket进行传输
    db.update_task(taskid, "completed", response_json)

    # 向前端发送消息, 结束
    #server.send_message(client, response_json)
    return response_json
    # 还需要写的()
    # 处理多个的部分
    # 文件接收模块
    # 线程安全模块
    # 测试一下后端安全模式
    # 还有前端, 差不多就这些了吧


# 当接收到消息时调用的回调函数

# 原函数名message_received
def message_handler(message):
    # print("server接收到消息:", message)
    try:
        # 解析 JSON 数据, 并得到对象名为data
        data = json.loads(message)
        type = data.get("type")

        print(type)

        # 传输的内容为seq
        if type == "seq":
            seq = data.get("body")
            # expose_sentence(seq, client, server, message)  # 执行单一的内容
            # 传入线程池执行对应函数
            pool.submit(expose_sentence, seq, message)
        # 传输的内容为文件
        elif type == "file":
            file_base64 = data.get("body")  # 获取64位文件编码
            filename = fileControl.save_base64_to_file(file_base64)  # 将其保存在File中, 并且返回文件名字
            # expose_fasta(filename, client, server, message)
            # 传入线程池执行对应的函数
            pool.submit(expose_fasta, filename, message)
            # 删除文件
            # fileControl.delete_file(filename)

        # 登录请求
        # 传入参数为(type, email)
        # 返回参数(type, exist, mes, info(userid,email))
        elif type == "login":

            # 准备相关内容
            email = data.get("email")
            exist, userid = db.user_existed(email)
            mes = ""
            if exist:
                mes = "OK"
            else:
                mes = "User does not"
                _, userid, _ = db.add_user(email)
            info = {"userid": userid, "email": email}

            # 构建信息体并且发送消息
            response_data = {
                "type": type,
                "exist": exist,
                "mes": mes,
                "info": info
            }
            print("登录请求的检查", response_data)
            response = json.dumps(response_data)  # json转化为字符串格式, 方便websocket进行传输
            # server.send_message(client, response)
            return response



        # 注册请求
        # 传入参数为(type, email)
        # 返回参数(type, mes, info(userid,email)) 返回用户的id
        elif type == "register":
            # 准备相关内容
            email = data.get("email")
            log, userid, notice = db.add_user(email)  # 找个log似乎查询不到
            info = {"userid": userid, "email": email}

            # 构建信息体并且发送消息
            response_data = {
                "type": type,
                "mes": notice,
                "info": info
            }
            print("注册的检查", response_data)
            response = json.dumps(response_data)  # json转化为字符串格式, 方便websocket进行传输
            # server.send_message(client, response)
            return response

        # 检查该用户下的所有任务序列
        # 传入参数(type, email)
        # 返回参数(type, email, tasks)
        elif type == "user_tasks":

            # 准备相关内容
            email = data.get("email")
            tasks = db.check_task(email)

            print("一共查询到了", len(tasks), "条任务")

            # 构建信息体并且发送消息
            response_data = {
                "type": type,
                "email": email,
                "tasks": tasks
            }
            response = json.dumps(response_data)  # json转化为字符串格式, 方便websocket进行传输
            # server.send_message(client, response)
            return response

        else:
            print("invalid request")

    # except ConnectionResetError:
    #    print(f"连接被重置，客户端 ID: {client['id']}")
    except json.JSONDecodeError:
        print("无效数据!")
    except:
        print("意外错误")
        return 400


# 创建 WebSocket 服务器

# 处理改为http,ws启用暂时关闭
'''
server = WebsocketServer(host='0.0.0.0', port=8080)
server.set_fn_new_client(new_client)
server.set_fn_message_received(message_received)
print("WebSocket 服务器正在运行，访问 ws://localhost:8080")
server.run_forever()
'''

# 上传文件以及seq的请求内容
"""
type: seq/file
user: 用户邮箱
body: seq字符串/base64文件
"""

# 返回消息的确定格式
"""
"message": "OK", //如果是ok才会携带这些东西
"Type": "single"/"mutil", # 代表单一的序列, 代表无论detail也好, meth也好, 都是针对单一seq的
"meth": json.dumps(container.meth),  # 发生了哪些修饰
"detail": json.dumps(container.detail) # 修饰的具体信息
"image": base,
"""

"""
任务状态:
'pending', 'in_progress', 'completed'
"""

# 以下是登录, 注册, 以及返回任务内容
"""
type:"login"
email:"xxxx"
"""

"""
type:"register"
email:"xxxxx"
"""

"""
type:"user_tasks"
email:"xxx"
"""

# 要想办法防止出问题, 全都放到try-catch里面

# 返回体:
"""
type:single/mutil/login/register/user_tasks 前面这些是单独的处理 notice是需要全局通报的特殊消息

"""
from flask_cors import CORS  # Import CORS
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
@app.route('/test', methods=['GET'])
def test():
    return 'Hello World'


@app.route('/api/req', methods=['POST'])
def handle_data():
    # 判断 content-type
    content_type = request.headers.get('Content-Type')

    if content_type == 'application/json':
        data = request.get_json()
        print(f"Received JSON: {data}")
        response = message_handler(message=json.dumps(data))
        if response == 400:
            return "发生问题", 400
        else:
            return response
    else:
        return 'Unsupported POST', 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
