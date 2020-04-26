import argparse
import os
import socket
import time

try:
    import _init_paths
except:
    pass
from watch_dog import get_free_gpu, get_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize weight of model and generate .table, weight/ and bias/")
    parser.add_argument("-i", action='store_true', help="Excute shell now.")
    return parser.parse_args()


def do_job(task_name, task, gpu_list, workers_per_gpu):
    assert len(gpu_list) > 0
    print("Executing shell...")
    if not task.startswith('configs'):
        task = 'configs/{}'.format(task)
    os.system(
        "./scripts/dist_train.sh {} {} {} {}".format(
            task, ','.join(gpu_list), task_name, workers_per_gpu))


def main():
    detected_times = 0
    task_id = 0
    first_run = True
    while True:
        if not first_run:
            time.sleep(10)
        first_run = False

        gpu_free_id, gpu_num = get_free_gpu(memory_thre)
        remove_id = []
        for gid in gpu_free_id:
            if int(gid) not in list(watch_gpus):
                remove_id.append(gid)
        for rid in remove_id:
            gpu_free_id.remove(rid)

        if len(gpu_free_id) > 0:
            detected_times += 1
            if len(gpu_free_id) == gpu_num:
                info_head = "All gpus detected"
            else:
                info_head = "Detect gpus: {}".format(gpu_free_id)
            print("{}, prepare to use, count {}.".format(
                info_head, counter - detected_times))
        else:
            detected_times = 0
            print("No free gpu detected.")
            continue

        if detected_times >= counter:
            if len(task_tuple) == task_id:
                print("Works have done!")
                exit(0)

            do_job(task_tuple[task_id][0], task_tuple[task_id][1], gpu_free_id, workers_per_gpu)
            task_id += 1

            if len(task_tuple) == task_id:
                print("Works have done!")
                exit(0)
            detected_times = counter - 6


if __name__ == '__main__':
    args = parse_args()
    hostname = socket.gethostname()
    cfg = get_config(hostname)

    task_tuple = [(cfg['tasks'][i]['NAME'], cfg['tasks'][i]['CFG'])
                  for i in range(len(cfg['tasks']))]
    watch_gpus = cfg.get('watch_gpus', list(range(8)))
    counter = int(cfg.get('wait_minutes', 0) * 6)  # we sleep 10s every loop
    memory_thre = cfg.get('memory_thre', 0.9)
    workers_per_gpu = cfg.get('workers_per_gpu', -1)

    if args.i:
        counter = 0
    main()
