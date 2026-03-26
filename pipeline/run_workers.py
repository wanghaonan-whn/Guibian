import time
from multiprocessing import Process, Queue

import toml
from loguru import logger

from pipeline.worker import save_worker, worker


def run_workers(config_path):
    workers = []
    save_workers = []
    config = toml.load(config_path)
    process_num = config["worker"]["process"]
    save_worker_config = config.get("save_worker", {})
    save_process_num = save_worker_config.get("process", max(1, process_num))
    save_queue_size = save_worker_config.get("queue_size", max(64, save_process_num * 8))
    start_time = time.time()
    device_ids = config["model"]["device"]
    save_queue = Queue(maxsize=save_queue_size)

    for rank in range(save_process_num):
        p = Process(target=save_worker, args=(config_path, save_queue, rank))
        p.start()
        save_workers.append(p)

    for rank, _ in enumerate(range(process_num)):
        p = Process(target=worker, args=(config_path, device_ids, rank, save_queue))
        p.start()
        workers.append(p)

    try:
        for p in workers:
            p.join()
    finally:
        for _ in save_workers:
            save_queue.put(None)
        for p in save_workers:
            p.join()
        logger.success("All save workers finished")

    end_time = time.time()
    logger.success(f"All workers finished")
    logger.debug(f"Total time: {end_time - start_time}")
