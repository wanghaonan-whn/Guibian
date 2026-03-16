import toml
from multiprocessing import Process
from pipeline.worker import worker


def run_workers(config_path):
    workers = []
    config = toml.load(config_path)
    process_num = config["model"]["device"]

    for rank, device_id in enumerate(process_num):
        p = Process(target=worker, args=(config_path, device_id, rank))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()
