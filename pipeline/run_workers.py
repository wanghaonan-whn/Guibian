import time

import toml
from multiprocessing import Process
from pipeline.worker import worker
from loguru import logger

def run_workers(config_path):
    workers = []
    config = toml.load(config_path)
    process_num = config["worker"]["process"]
    start_time = time.time()
    for rank, device_id in enumerate(process_num):
        p = Process(target=worker, args=(config_path, device_id, rank))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()
    end_time = time.time()
    logger.success(f"All workers finished")
    logger.debug(f"Total time: {end_time - start_time}")