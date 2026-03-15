import subprocess
import time
from multiprocessing import Process


def start_receiver():
    """ 启动 HTTP 接收服务 """
    subprocess.run(["python", "receiver.py"])


def start_workers():
    """ 启动 worker 进程 """
    subprocess.run(["python", "run_workers.py"])


def main():
    # 启动 worker
    worker_process = Process(target=start_workers)
    worker_process.start()

    # 等待 worker 启动
    time.sleep(1)

    # 启动 receiver
    receiver_process = Process(target=start_receiver)
    receiver_process.start()

    worker_process.join()
    receiver_process.join()


if __name__ == "__main__":
    main()
