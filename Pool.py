import threading
import queue
import traceback

class SimpleThreadPool:
    def __init__(self, num_threads=4):
        self.num_threads = num_threads
        self.task_queue = queue.Queue()
        self.threads = []
        self._shutdown = False

        # 启动工作线程
        for _ in range(num_threads):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.threads.append(t)

    def _worker(self):
        """工作线程主循环"""
        while not self._shutdown:
            try:
                # 获取任务（带超时避免永久阻塞）
                task = self.task_queue.get(timeout=1)

                # 执行任务
                func, args, kwargs = task
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"任务执行失败: {e}\n{traceback.format_exc()}")
                finally:
                    self.task_queue.task_done()

            except queue.Empty:  # 空队列时重试
                continue
            except Exception as e:  # 其他异常处理
                print(f"工作线程异常: {e}\n{traceback.format_exc()}")
                break

    def submit(self, func, *args, **kwargs):
        """提交任务（保持原接口）"""
        print("测试")
        self.task_queue.put((func, args, kwargs))

    def close(self):
        """安全关闭线程池"""
        # 标记关闭状态
        self._shutdown = True

        # 等待队列任务完成
        self.task_queue.join()

        # 等待线程自然退出
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=1)

        print("所有线程已安全退出")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

"""
想使用的话就是这样的
pool = SimpleProcessPool(num_processes=2)
pool.submit(worker_task, 1)
"""