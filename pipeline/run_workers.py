from multiprocessing import Process
from worker import worker


def run():
    workers = []
    for _ in range(9):   # 4个处理进程
        p = Process(target=worker)
        p.start()
        workers.append(p)

    for p in workers:
        p.join()
