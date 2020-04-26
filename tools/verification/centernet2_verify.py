import os
import numpy as np
import random
import torch
import cv2
import math
import pickle as pkl
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, raw_inference
from mmdet.core.utils.model_utils import describe_vars
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from vis_util import show_corners, draw_polygon
from tools.model_zoo import model_zoo as zoo

version = 'v5.4.2'
cfg = mmcv.Config.fromfile(zoo[version]['config'])
cfg.model.pretrained = None

print('loading model...')
# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)

model.load_state_dict(
    torch.load(zoo[version]['model_file'])['state_dict'], strict=False)
for m in model.modules():
    m.eval()
model.eval()
model.cuda()


def get_imlist():
    data_dir = '/private/ningqingqun/bags/truck5_2019-05-15-15-46-14_2.bag/mid_left/201905151546'
    data_dir = '/private/ningqingqun/bags/truck1_2019_09_06_13_48_14_19.msg/front_right/201909061348'
    # data_dir = '/private/ningqingqun/bags/truck1_2019_07_24_14_54_43_26.msg/front_right/201907241455'
    # data_dir = '/private/ningqingqun/bags/truck1_2019_08_01_15_21_03_19.msg/front_right/201908011521'
    # data_dir = '/nfs/lidar_data_v2/TRUCK_4/2019_05_16/1040/0_10.0_30.0/camera_front_right'
    out_dir = '/private/ningqingqun/results/centernet2_results/' + data_dir.split(
        '/')[-1]
    print('results will save to {}'.format(out_dir))
    im_list = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('.jpg')
    ]

    return im_list, out_dir


def get_imlist2():
    list_file = '/private/ningqingqun/obstacle_detector/data/obstacle2d/ImageSets/val1010.txt'
    im_prefix = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages/'
    with open(list_file) as f:
        imlist = f.read().splitlines()
    imlist = [os.path.join(im_prefix, l + '.jpg') for l in imlist]

    out_dir = '/private/ningqingqun/results/centernet2_results/val1010'
    print('results will save to {}'.format(out_dir))

    return imlist, out_dir


def preprocess(image, use_rgb=True):
    image = image.astype(np.float32)
    mean_rgb = np.array([123.675, 116.28, 103.53])
    std_rgb = np.array([58.395, 57.12, 57.375])

    if use_rgb:
        image = image[..., [2, 1, 0]]
        image -= mean_rgb
        image /= std_rgb
    else:
        mean_bgr = mean_rgb[[2, 1, 0]]
        std_bgr = std_rgb[[2, 1, 0]]
        image -= mean_bgr
        image /= std_bgr

    image = np.transpose(image, [2, 0, 1])
    image = torch.tensor(image)
    image = image.view(1, *image.size())

    return image


pose2closed = {
    0: [3, 5],
    1: [4, 5],
    2: [3, 5],
    3: [3, 4],
    4: [3, 5],
    5: [4, 5],
    6: [3, 5],
    7: [3, 4],
}


def get_det_bboxes(results, score_thr=0.1):
    corners, labels, scores, poses = results
    bboxes = []
    inds = np.where(scores > score_thr)[0]
    for i in inds.tolist():
        corners_per_object = corners[i]
        x1 = corners_per_object[[0, 2, 3, 5], 0].min()
        x2 = corners_per_object[[0, 2, 3, 5], 0].max()
        y1 = corners_per_object[[0, 2, 3, 5], 1].min()
        y2 = corners_per_object[[0, 2, 3, 5], 1].max()
        bboxes.append([x1, y1, x2, y2])

    return np.array(bboxes, dtype=np.float32)


def verify_images():
    im_list, out_dir = get_imlist()
    print('detect {} images ...'.format(len(im_list)))
    for imf in tqdm(im_list, total=len(im_list)):
        im = cv2.imread(imf)
        reiszed_im = cv2.resize(im, (1280, 800))
        inputs = preprocess(reiszed_im)

        results = model(inputs.cuda(), return_loss=False, try_dummy=False)
        out_file = os.path.join(out_dir, os.path.basename(imf))
        show_corners(
            reiszed_im,
            results,
            class_names,
            score_thr=thresh,
            out_file=out_file,
            pad=250)

        # corners, labels, scores, poses = results
        # scale = np.array(im.shape[:2]) / np.array(reiszed_im.shape[:2])
        # corners_scale = corners * scale[[1, 0]]
        # corners_pairs = []
        # for i in range(len(corners_scale)):
        #     if scores[i] < thresh:
        #         continue
        #     corners_per_object = corners_scale[i]
        #     corner_idx = pose2closed[poses[i]]
        #     pair = corners_per_object[corner_idx]
        #     corners_pairs.append('{:.2f} {:.2f} {:.2f} {:.2f}'.format(
        #         pair[0, 0], pair[0, 1], pair[1, 0], pair[1, 1]))

        # txt_file = os.path.join(
        #     '/private/ningqingqun/results/centernet2_results/201909061348_txt',
        #     imf.split('_')[-1].replace('.jpg', '.txt'))
        # with open(txt_file, 'w') as f:
        #     f.write('\n'.join(corners_pairs))


def verify_video():
    out_dir = '/private/ningqingqun/results/senyun'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    vfile = '/private/ningqingqun/datasets/videos/WIN_20191030_20_02_53_Pro60.mp4'
    cap = cv2.VideoCapture(vfile)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(total_frame)

    display_num = 20
    base_frame_index = random.randint(0, total_frame - display_num)

    frame_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, base_frame_index)
    while (cap.isOpened()):
        ret, frame = cap.read()
        reiszed_im = cv2.resize(frame, (1280, 800))
        inputs = preprocess(reiszed_im)

        result = model(inputs.cuda(), return_loss=False, try_dummy=False)
        out_file = os.path.join(
            out_dir, '{:04}.jpg'.format(base_frame_index + frame_count))
        show_corners(
            reiszed_im,
            result,
            class_names,
            score_thr=thresh,
            out_file=out_file,
            pad=250)

        frame_count += 1
        if frame_count > display_num:
            break
    cap.release()


def verify_split():
    data_root = 'data/fabu3d'
    split = 'val_adc6f80f'
    out_dir = '/private/ningqingqun/results/centernet2_results/' + split
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    split_file = os.path.join(data_root, 'pkl', split + '.pkl')
    dataset = pkl.load(open(split_file, 'rb'), encoding='latin1')
    for _, label in tqdm(dataset.items()):
        imf = os.path.join(data_root, 'images', label['filename'])
        im = cv2.imread(imf)
        reiszed_im = cv2.resize(im, (1280, 800))
        gt_im = reiszed_im.copy()
        inputs = preprocess(reiszed_im)

        results = model(inputs.cuda(), return_loss=False, try_dummy=False)
        out_file = os.path.join(out_dir, os.path.basename(imf))
        predict_im = show_corners(
            reiszed_im, results, class_names, score_thr=thresh, pad=250)

        corners = label['ann']['corners']
        poses = label['ann']['poses']
        for i in range(len(poses)):
            vcorners = (corners[i] / 1.5).astype(np.int32).reshape(-1, 1, 2)
            draw_polygon(gt_im, vcorners, poses[i])

        gt_im = cv2.resize(gt_im, (predict_im.shape[1], predict_im.shape[0]))
        display_im = np.vstack([predict_im, gt_im])
        cv2.imwrite(out_file, display_im)


def verify_match():
    data_root = 'data/fabu3d'
    out_dir = '/private/ningqingqun/results/centernet2_results/match'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    split_file = os.path.join(data_root, 'pkl', 'val.pkl')
    dataset = pkl.load(open(split_file, 'rb'), encoding='latin1')
    for _, label in tqdm(dataset.items()):
        imf = os.path.join(data_root, 'images', label['filename'])
        im = cv2.imread(imf)
        reiszed_im = cv2.resize(im, (1280, 800))
        gt_im = reiszed_im.copy()
        inputs = preprocess(reiszed_im)

        results = model(inputs.cuda(), return_loss=False, try_dummy=False)
        out_file = os.path.join(out_dir, os.path.basename(imf))
        det_bboxes = get_det_bboxes(results)
        gt_bboxes = label['ann']['bboxes'] / 1.5
        ious = bbox_overlaps(det_bboxes, gt_bboxes)
        ious_argmax = ious.argmax(axis=1)
        ious_max = ious.max(axis=1)

        rand_colors = np.random.rand(100, 3) * 255
        # draw det bboxes
        for i in range(len(det_bboxes)):
            b = det_bboxes[i]
            if ious_max[i] > 0.2:
                color = rand_colors[ious_argmax[i]]
            else:
                color = (0, 0, 0)
            cv2.rectangle(
                reiszed_im, (b[0], b[1]), (b[2], b[3]), color, thickness=2)

        # draw gt bboxes
        for i in range(len(gt_bboxes)):
            b = gt_bboxes[i]
            color = rand_colors[i]
            cv2.rectangle(
                gt_im, (b[0], b[1]), (b[2], b[3]), color, thickness=2)

        display_im = np.vstack([reiszed_im, gt_im])
        cv2.imwrite(out_file, display_im)


thresh = 0.4
class_names = ['car', 'bus', 'truck', 'person', 'bicycle', 'tricycle', 'block']

verify_images()
