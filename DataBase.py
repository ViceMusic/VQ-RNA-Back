"""
author: XarNud Vilas

数据库的控制模块, 提供一些函数, 使用对象类进行封装
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from Utils import lock


class Database:
    """SQLite 数据库操作类 (线程安全连接)"""
    def __init__(self, db_name: str = "vq-rna.db"):
        # 确保数据库文件存在于当前目录
        self.db_path = Path(__file__).parent / db_name
        self.connection = None
        self._connect()

    def _connect(self):
        """建立数据库连接（自动创建文件）"""
        try:
            self.connection = sqlite3.connect(
                str(self.db_path),
                timeout=10,  # 避免多线程竞争
                isolation_level=None,  # 手动控制事务
                check_same_thread=False  # 允许多线程
            )
            self.connection.execute("PRAGMA journal_mode=WAL")  # 提高并发性能
            self.connection.execute("PRAGMA foreign_keys=ON")  # 启用外键约束
        except sqlite3.Error as e:
            raise RuntimeError(f"数据库连接失败: {e}")

    # sql语句的执行方法
    def execute(
            self,
            sql: str,
            params: Optional[tuple] = None,
            *,
            commit: bool = True
    ) -> sqlite3.Cursor:
        """
        执行SQL语句
        :param sql: SQL语句
        :param params: 参数元组
        :param commit: 是否自动提交
        :return: 游标对象
        """
        with lock: # 每次进行对应操作的时候, 都进行上锁以防止发生问题
            try:
                cursor = self.connection.cursor()
                cursor.execute(sql, params or ())
                if commit:
                    self.connection.commit()
                return cursor
            except sqlite3.Error as e:
                self.connection.rollback()
                raise RuntimeError(f"SQL执行错误: {e}\nSQL: {sql}")

    def fetch_all(self, sql: str, params: Optional[tuple] = None) :
        """查询多条记录（返回字典列表）"""
        cursor = self.execute(sql, params, commit=False)
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def user_existed(self, email: str) :
        sql = "SELECT userid FROM users WHERE email = ?"
        result = self.fetch_all(sql, (email,))
        if len(result) > 0:
            return True, result[0]["userid"]
        else:
            return False, None

    def task_existed(self, id : int) :
        sql = "SELECT taskid FROM tasks WHERE taskid = ?"
        result = self.fetch_all(sql, (id,))
        return len(result) > 0

    def add_user(self, email: str) :
        exist, _ =self.user_existed(email)
        if exist:
            return False, None, "邮箱已存在"
        try:
            sql = "INSERT INTO users (email) VALUES (?)"
            cursor=self.execute(sql, (email,))
            print(cursor)
            print("成功新增用户")
            return True, cursor.lastrowid, "获取用户成功"
        except sqlite3.Error as e:
            print(f"添加用户失败: {e}")
            return False, None, f"添加用户失败: {e}"

    def check_task(self, email: str) :
        sql = "SELECT * FROM tasks WHERE userid in (SELECT userid from users WHERE email = ?)"
        cursor = self.execute(sql, (email,))
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def update_task(self,taskid : int, state : str, content : str) :
        if self.task_existed(taskid):
            sql = "UPDATE tasks SET state = ?, content = ? WHERE taskid = ?"
            self.execute(sql, (state, content, taskid))
            return "更新完成"
        else :
            return "任务不存在"

    def add_task(self, userid: int, content: str, state :str) :
        try:
            start_time = datetime.now()
            sql = "INSERT INTO tasks (userid, state, content, startTime) VALUES (?, ?, ?, ?)"
            cursor=self.execute(sql, (userid, state, content, start_time))
            taskid = cursor.lastrowid
            return True, taskid, "任务添加成功"
        except sqlite3.Error as e:
            return False, f"添加任务失败: {e}"

    def delete_task(self, taskid: int) :
        if self.task_existed(taskid):
            sql = "DELETE FROM tasks WHERE taskid = ?"
            self.execute(sql, (taskid,))
            return "删除完成"
        else:
            return "任务不存在"




# 使用这个函数就会返回具体的任务信息
#success,taskid,message=db.add_task("1","内容","pending")
#print(taskid)

#根据邮箱查询用户id
#mes,userid=db.user_existed("user1@example.com")
#print(userid)

#根据用户id, 当前内容等等新增内容


"""
Notice:

数据库设计的有问题, 理论上userId应该是唯一主键
但是有的时候检查用户是通过userID, 有时候是通过email检查, 有点毛病
"""

