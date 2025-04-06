# 负责文件管理类型, 负责上传文件, 删除文件, 等等
import os
import uuid
import base64
from Utils import generate_unique_filename,lock

class FileControl:
    #初始化文件夹
    def __init__(self, base_dir="Files"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
    #将二进制数据保存在文件夹中

    def save_base64_to_file(self, base64_str: str, filename: str = None) -> str:
        try:
            #为文件起名
            filename = generate_unique_filename(extension=".fasta")
            file_path = os.path.join(self.base_dir, filename)
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(base64_str.encode('utf-8')))
            return filename
        except Exception as e:
            raise ValueError(f"Base64保存失败: {str(e)}")

    def find_file(self, filename: str) -> str:
        """查找文件返回完整路径"""
        file_path = os.path.join(self.base_dir, filename)
        if os.path.exists(file_path):
            return file_path
        else:
            return None


    def delete_file(self, filename: str) -> None:
        with lock:
            # 安全检查：防止路径穿越攻击
            if os.path.sep in filename or filename.startswith('.') or '..' in filename:
                raise ValueError("文件名不能包含路径")

            file_path = os.path.join(self.base_dir, filename)

            if not os.path.exists(file_path):
                raise ValueError(f"文件不存在: {filename}")

            if not os.path.isfile(file_path):
                raise ValueError(f"不是文件: {filename}")

            os.remove(file_path)

# 对文件夹的操作信息



'''
现在还剩下:
1. 模型预测更新一下
2. 对Fastq的文件支持
3. 对文件的画图支持
4. 前端联动后端
'''


# fc.delete_file("file_1.bin") 差不多就是这种操作

