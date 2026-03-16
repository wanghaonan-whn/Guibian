import argparse
import time
from multiprocessing import Process
from pipeline.run_workers import run_workers
from service.receiver import run_receiver


def parse_args():
    parser = argparse.ArgumentParser(prog="main.py")
    parser.add_argument("--config-path", default="./conf/config_whn.yaml", type=str)
    return parser.parse_args()


def main():
    # 启动 worker
    args = parse_args()
    config_path = args.config_path
    worker_process = Process(target=run_workers, args=(config_path,))
    worker_process.start()

    # 等待 worker 启动
    time.sleep(1)

    # 启动 receiver
    receiver_process = Process(target=run_receiver)
    receiver_process.start()

    worker_process.join()
    receiver_process.join()


if __name__ == "__main__":
    main()
