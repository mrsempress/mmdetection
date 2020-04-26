# This file is for train self-supervised model, and it inherits the train.py.
"""
CUDA_VISIBLE_DEVICES=1,2,3 python3 tools/train_ssl.py configs/ssl/r50_moco.py --dir=MoCo --launcher pytorch
"""
from __future__ import division

try:
    import _init_paths
except:
    pass

import argparse
import os
import time
import shutil

import torch
from mmcv import Config
# from mmcv.runner import init_dist
from mmcv.runner import get_dist_info
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel

from mmdet import __version__
from mmdet.core.utils import logger as nlogger
from mmdet.core.utils.summary import set_writer
from mmdet.core.utils.model_utils import describe_vars
from mmdet.apis import get_root_logger, set_random_seed, train_detector, init_dist
from mmdet.datasets import build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--dir', help='the dir to save logs and models')
    parser.add_argument(
        '--workers_per_gpu',
        type=int,
        default=-1,
        help='worker num per gpu')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--test', action='store_true', help='replace the train dataset to speed up')
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the batch size')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.dir is not None:
        if args.dir.startswith('//'):
            cfg.work_dir = args.dir[2:]
        else:
            if args.dir.endswith('-c'):
                args.dir = args.dir[:-2]
                args.resume_from = search_latest_checkpoint(os.path.join('work_dirs', args.dir))
            cfg.work_dir += time.strftime("_%m%d_%H%M")
            cfg.work_dir = os.path.join('work_dirs', args.dir, cfg.work_dir)

    if args.workers_per_gpu != -1:
        cfg.data['workers_per_gpu'] = args.workers_per_gpu

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if cfg.resume_from or cfg.load_from:
        cfg.model['pretrained'] = None

    if args.test:
        cfg.data.train['ann_file'] = cfg.data.val['ann_file']
        cfg.data.train['img_prefix'] = cfg.data.val['img_prefix']

    # init distributed env first, since logger depends on the dist info.
    distributed = True
    init_dist(args.launcher, **cfg.dist_params)
    num_gpus = torch.cuda.device_count()
    rank, _ = get_dist_info()

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] *= ((num_gpus / 8) * (cfg.data.imgs_per_gpu / 2))

    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_root_logger(nlogger, cfg.log_level)
    if rank == 0:
        logger.set_logger_dir(cfg.work_dir, 'd')
    logger.info("Config: ------------------------------------------\n" + cfg.text)
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('MMDetection Version: {}'.format(__version__))
    logger.info('Config:\n{}'.format(cfg.text))

    # set random seeds
    if hasattr(cfg, 'seed'):
        args.seed = cfg.seed
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed, deterministic=args.deterministic)

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if rank == 0:
        writer = set_writer(cfg.work_dir)
        if args.seed is None:  # forward may affect the random in each GPU
            describe_vars(model)

    model = MMDistributedDataParallel(model.cuda())

    runner_attr_dict = {'task_name': args.dir}
    datasets = [build_dataset(cfg.data.train)]

    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text)
    # add an attribute for visualization convenience
    # if hasattr(model, 'module'):
    #     model.module.CLASSES = datasets[0].CLASSES
    # else:
    #     model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=False,
        logger=logger,
        timestamp=timestamp,
        runner_attr_dict=runner_attr_dict)


def clear_work_dirs(soft_clear=True):
    dirname = nlogger.get_logger_dir()
    if dirname and os.path.isdir(dirname):
        for f in os.listdir(dirname):
            if f.endswith('.pth') or os.path.isdir(os.path.join(dirname, f)):
                return

        if soft_clear:
            time.sleep(1)
            action = input("Select Action: k (keep) / d (delete):").lower().strip()
            act = action
            if act == 'd':
                shutil.rmtree(dirname, ignore_errors=True)
        else:
            shutil.rmtree(dirname, ignore_errors=True)


def search_latest_checkpoint(dir_name):
    latest_checkpoint = None
    latest_checkpoint_dir = None
    latest_epoch = 0
    latest_iter = 0

    for root, dir, files in os.walk(dir_name):
        for name in files:
            file_path = root + "/" + name
            if not file_path.endswith('.pth') or 'latest' in file_path:
                continue

            checkpoint_name_split = name[:-4].split('_')
            if int(checkpoint_name_split[1]) > latest_epoch:
                latest_epoch = int(checkpoint_name_split[1])
                latest_checkpoint_dir = root
            if len(checkpoint_name_split) == 4 and int(checkpoint_name_split[3]) > latest_iter:
                latest_iter = int(checkpoint_name_split[3])
                latest_checkpoint_dir = root

    if latest_checkpoint_dir is not None:
        if os.path.isfile(os.path.join(latest_checkpoint_dir,
                                       "epoch_{}".format(latest_epoch))):
            latest_checkpoint = os.path.join(latest_checkpoint_dir,
                                             "epoch_{}.pth".format(latest_epoch))
        else:
            latest_checkpoint = os.path.join(latest_checkpoint_dir,
                                             "epoch_{}_iter_{}.pth".format(latest_epoch,
                                                                           latest_iter))
    return latest_checkpoint


if __name__ == '__main__':
    start = time.time()
    try:
        main()
    except KeyboardInterrupt:
        if time.time() - start < 300:
            clear_work_dirs(soft_clear=True)
    except Exception as e:
        if time.time() - start < 300:
            clear_work_dirs(soft_clear=False)
        raise e
    finally:
        torch.cuda.empty_cache()
