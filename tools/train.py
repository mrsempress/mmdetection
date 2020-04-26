from __future__ import division

try:
    import _init_paths
except:
    pass

import argparse
import os
import time
import shutil
import mmcv
from mmcv import Config
from mmcv.runner import get_dist_info
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel

from mmdet import __version__
from mmdet.core.utils import logger as nlogger
from mmdet.core.utils.misc import get_localhost
from mmdet.core.utils.summary import set_writer
from mmdet.core.utils.model_utils import describe_vars, describe_features
from mmdet.datasets import build_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--dir', help='the dir to save logs and models')
    parser.add_argument(
        '--workers_per_gpu', type=int, default=-1, help='worker num per gpu')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--graph', action='store_true', help='grad graph')
    parser.add_argument(
        '--profiler', choices=['train', 'test'], help='profiler')
    parser.add_argument('--speed', choices=['train', 'test'], help='get speed')
    parser.add_argument(
        '--test',
        action='store_true',
        help='replace the train dataset to speed up')
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
            localhost = get_localhost().split('.')[0]
            # results from server saved to /private
            if 'gpu' in localhost:
                output_dir = '/private/huangchenxi/mmdet/outputs'
            else:
                output_dir = 'work_dirs'

            if args.dir.endswith('-c'):
                args.dir = args.dir[:-2]
                args.resume_from = search_and_delete(
                    os.path.join(output_dir, args.dir),
                    prefix=cfg.work_dir,
                    suffix=localhost)
            cfg.work_dir += time.strftime("_%m%d_%H%M") + '_' + localhost
            cfg.work_dir = os.path.join(output_dir, args.dir, cfg.work_dir)

    if args.workers_per_gpu != -1:
        cfg.data['workers_per_gpu'] = args.workers_per_gpu

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.profiler or args.speed:
        cfg.data.imgs_per_gpu = 1

    if cfg.resume_from or cfg.load_from:
        cfg.model['pretrained'] = None

    if args.test:
        cfg.data.train['ann_file'] = cfg.data.val['ann_file']
        cfg.data.train['img_prefix'] = cfg.data.val['img_prefix']

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        num_gpus = args.gpus
        rank = 0
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        num_gpus = torch.cuda.device_count()
        rank, _ = get_dist_info()

    if cfg.optimizer['type'] == 'SGD':
        cfg.optimizer['lr'] *= num_gpus * cfg.data.imgs_per_gpu / 256
    else:
        cfg.optimizer['lr'] *= ((num_gpus / 8) * (cfg.data.imgs_per_gpu / 2))

    # init logger before other steps
    logger = get_root_logger(nlogger, cfg.log_level)
    if rank == 0:
        logger.set_logger_dir(cfg.work_dir, 'd')
    logger.info("Config: ------------------------------------------\n" +
                cfg.text)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if rank == 0:
        # describe_vars(model)
        writer = set_writer(cfg.work_dir)
        # try:
        #     # describe_features(model.backbone)
        #     writer.add_graph(model, torch.zeros((1, 3, 800, 800)))
        # except (NotImplementedError, TypeError):
        #     logger.warn("Add graph failed.")
        # except Exception as e:
        #     logger.warn("Add graph failed:", e)

    if not args.graph and not args.profiler and not args.speed:
        if distributed:
            model = MMDistributedDataParallel(model.cuda())
        else:
            model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

        if isinstance(cfg.data.train, list):
            for t in cfg.data.train:
                logger.info("loading training set: " + str(t.ann_file))
            train_dataset = [build_dataset(t) for t in cfg.data.train]
            CLASSES = train_dataset[0].CLASSES
        else:
            logger.info("loading training set: " +
                        str(cfg.data.train.ann_file))
            train_dataset = build_dataset(cfg.data.train)
            logger.info("{} images loaded!".format(len(train_dataset)))
            CLASSES = train_dataset.CLASSES
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__, config=cfg.text, CLASSES=CLASSES)
        # add an attribute for visualization convenience
        if hasattr(model, 'module'):
            model.module.CLASSES = CLASSES
        else:
            model.CLASSES = CLASSES
        train_detector(
            model,
            train_dataset,
            cfg,
            distributed=distributed,
            validate=args.validate,
            logger=logger,
            runner_attr_dict={'task_name': args.dir})
    else:
        from mmcv.runner.checkpoint import load_checkpoint
        from mmdet.datasets import build_dataloader
        from mmdet.core.utils.model_utils import register_hooks
        from mmdet.apis.train import parse_losses

        model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
        if args.profiler == 'test' or args.speed == 'test':
            model.eval()
            dataset = build_dataset(cfg.data.test)
        else:
            model.train()
            dataset = build_dataset(cfg.data.train)

        if cfg.load_from and (args.profiler or args.speed):
            logger.info('load checkpoint from %s', cfg.load_from)
            load_checkpoint(
                model, cfg.load_from, map_location='cpu', strict=True)

        data_loader = build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False,
            shuffle=False)

        if args.graph:
            id_dict = {}
            for name, parameter in model.named_parameters():
                id_dict[id(parameter)] = name

        for i, data_batch in enumerate(data_loader):
            if args.graph:
                outputs = model(**data_batch)
                loss, log_vars = parse_losses(outputs)
                get_dot = register_hooks(loss, id_dict)
                loss.backward()
                dot = get_dot()
                dot.save('graph.dot')
                break
            elif args.profiler:
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    if args.profiler == 'train':
                        outputs = model(**data_batch)
                        loss, log_vars = parse_losses(outputs)
                        loss.backward()
                    else:
                        with torch.no_grad():
                            model(**data_batch, return_loss=False)

                    if i == 20:
                        prof.export_chrome_trace('./trace.json')
                        logger.info(prof)
                        break
            elif args.speed:
                if args.speed == 'train':
                    start = time.perf_counter()
                    outputs = model(**data_batch)
                    loss, log_vars = parse_losses(outputs)
                    loss.backward()
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                else:
                    start = time.perf_counter()
                    with torch.no_grad():
                        model(**data_batch, return_loss=False)
                    end = time.perf_counter()
                logger.info("{:.3f} s/iter, {:.1f} iters/s".format(
                    end - start, 1. / (end - start)))


def clear_work_dirs(soft_clear=True):
    dirname = nlogger.get_logger_dir()
    if dirname and os.path.isdir(dirname):
        for f in os.listdir(dirname):
            if f.endswith('.pth') or os.path.isdir(os.path.join(dirname, f)):
                return

        if soft_clear:
            time.sleep(1)
            action = input(
                "Select Action: k (keep) / d (delete):").lower().strip()
            act = action
            if act == 'd':
                shutil.rmtree(dirname, ignore_errors=True)
        else:
            shutil.rmtree(dirname, ignore_errors=True)


def search_and_delete(dir_name, delete_empty=True, prefix='', suffix=''):
    exp_list = [os.path.join(dir_name, f) for f in os.listdir(dir_name)]
    exp_list = [
        d for d in exp_list if os.path.isdir(d) and suffix in d and prefix in d
    ]
    exp_list.sort(reverse=True)
    latest_checkpoint = None
    rank = get_dist_info()[0]
    for d in exp_list:
        checkpoint_list = [
            os.path.join(d, f) for f in os.listdir(d)
            if f.endswith('.pth') and f.startswith('epoch')
        ]
        latest_epoch = 0
        for f in checkpoint_list:
            epoch = os.path.basename(f[:-4]).split('_')[1]
            if int(epoch) > latest_epoch:
                latest_epoch = int(epoch)
                latest_checkpoint = f

        if latest_checkpoint is not None:
            break
        elif delete_empty and rank == 0:
            shutil.rmtree(d, ignore_errors=True)

    return latest_checkpoint


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
            if len(checkpoint_name_split) == 4 and int(
                    checkpoint_name_split[3]) > latest_iter:
                latest_iter = int(checkpoint_name_split[3])
                latest_checkpoint_dir = root

    if latest_checkpoint_dir is not None:
        if os.path.isfile(
                os.path.join(latest_checkpoint_dir,
                             "epoch_{}".format(latest_epoch))):
            latest_checkpoint = os.path.join(
                latest_checkpoint_dir, "epoch_{}.pth".format(latest_epoch))
        else:
            latest_checkpoint = os.path.join(
                latest_checkpoint_dir,
                "epoch_{}_iter_{}.pth".format(latest_epoch, latest_iter))
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
