import argparse
import os
import os.path as osp
import shutil
import tempfile
import numpy as np

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import (results2json, coco_eval, wrap_fp16_model, eval_map,
                        eval_corners)
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(
                data, result, dataset.img_norm_cfg, show=True)

            if i == 20:
                break

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def corners2bboxes(outputs, num_fg):
    num_images = len(outputs)
    bboxes = []
    for i in range(num_images):
        if len(outputs[i]) == 5:
            corners_per_image, labels_per_image, scores_per_image, _, _ = outputs[
                i]
        else:
            corners_per_image, labels_per_image, scores_per_image, _ = outputs[
                i]
        if corners_per_image.shape[0] == 0:
            bboxes_per_class = [
                np.zeros((0, 5), dtype=np.float32) for i in range(num_fg)
            ]
        else:
            x1 = corners_per_image[:, :, 0].min(axis=1)
            x2 = corners_per_image[:, :, 0].max(axis=1)
            y1 = corners_per_image[:, :, 1].min(axis=1)
            y2 = corners_per_image[:, :, 1].max(axis=1)
            bboxes_per_image = np.stack([x1, y1, x2, y2, scores_per_image],
                                        axis=1)
            bboxes_per_class = [
                bboxes_per_image[labels_per_image == c, :]
                for c in range(num_fg)
            ]
        bboxes.append(bboxes_per_class)

    return bboxes


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file', default=None)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=[
            'proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints', 'corners'
        ],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    if args.out is None:
        dataset_name = dataset.name if hasattr(dataset, 'name') else 'val'
        if hasattr(cfg.data.test, 'task'):
            dataset_name = dataset_name + '_' + cfg.data.test.task
        model_name = os.path.basename(args.checkpoint).split('.')[0]
        model_dir = os.path.dirname(args.checkpoint)
        args.out = os.path.join(model_dir, 'raw_results',
                                dataset_name + '_' + model_name + '.pkl')
    elif not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    mmcv.mkdir_or_exist(os.path.dirname(args.out))

    rank, _ = get_dist_info()
    eval_types = args.eval
    if not os.path.isfile(args.out):
        # build the model and load checkpoint
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(
            model, args.checkpoint, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show)
        else:
            model = MMDistributedDataParallel(model.cuda())
            outputs = multi_gpu_test(model, data_loader, args.tmpdir)

        if rank == 0:
            if hasattr(dataset, 'raw_annotations'):
                filenames = [
                    dataset.raw_annotations[dataset.ids[i]]['filename']
                    for i in range(len(dataset))
                ]
            else:
                filenames = [
                    img_info['filename'] for img_info in dataset.img_infos
                ]

            print('\nwriting results to {}'.format(args.out))
            results = {
                'file_names': filenames,
                'outputs': outputs,
            }
            mmcv.dump(results, args.out, protocol=2)
    elif rank == 0:
        results = mmcv.load(args.out, encoding='latin1')
        outputs = results['outputs']

    if eval_types and rank == 0:
        print('Starting evaluate {}'.format(' and '.join(eval_types)))
        if not hasattr(dataset, 'coco'):
            if hasattr(dataset, 'raw_annotations'):
                gt_bboxes = [
                    dataset.raw_annotations[dataset.ids[i]]['ann']['bboxes']
                    for i in range(len(dataset))
                ]
                gt_labels = [
                    dataset.raw_annotations[dataset.ids[i]]['ann']['classes']
                    for i in range(len(dataset))
                ]

                if cfg.data.test.with_ignore:
                    gt_ignores = [l <= 0 for l in gt_labels]
                else:
                    gt_ignores = [l == 0 for l in gt_labels]
                gt_labels = [np.abs(l) for l in gt_labels]
                if 'corners' in eval_types:
                    gt_corners = [
                        dataset.raw_annotations[dataset.ids[i]]['ann']
                        ['corners'] for i in range(len(dataset))
                    ]
                    gt_poses = [
                        dataset.raw_annotations[dataset.ids[i]]['ann']['poses']
                        for i in range(len(dataset))
                    ]
                    eval_corners(
                        outputs,
                        gt_corners,
                        gt_poses,
                        gt_labels,
                        gt_ignores,
                        gt_bboxes=gt_bboxes,
                        display=True)
                    det_bboxes = corners2bboxes(outputs,
                                                len(dataset.CLASSES) - 1)
                    eval_map(
                        det_bboxes, gt_bboxes, gt_labels, gt_ignore=gt_ignores)
                else:
                    eval_map(
                        outputs, gt_bboxes, gt_labels, gt_ignore=gt_ignores)
            else:
                gt_bboxes = [
                    img_info['ann']['bboxes'] for img_info in dataset.img_infos
                ]
                gt_labels = [
                    img_info['ann']['labels'] for img_info in dataset.img_infos
                ]
                if len(outputs[0]) == 5:
                    outputs = corners2bboxes(outputs, len(dataset.classes) - 1)
                eval_map(outputs, gt_bboxes, gt_labels, iou_thr=0.4)
        else:
            if eval_types == ['proposal_fast']:
                result_file = args.out
                coco_eval(result_file, eval_types, dataset.coco, CLASSES=dataset.CLASSES)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json(dataset, outputs, args.out)
                    coco_eval(result_files, eval_types, dataset.coco, CLASSES=dataset.CLASSES, show=True)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file)
                        coco_eval(result_files, eval_types, dataset.coco, CLASSES=dataset.CLASSES, show=True)


if __name__ == '__main__':
    main()
