import os
import numpy as np
import random
import torch
import cv2
import math
import random
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import pickle as pkl

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, raw_inference
from mmdet.core.utils.model_utils import describe_vars
from mmdet.ops import nms
from tools.model_zoo import model_zoo as zoo

task = 'v4.2.1'
cfg_file = zoo[task]['config']
cfg = mmcv.Config.fromfile(cfg_file)
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
model_file = zoo[task]['model_file']
print('loading from ' + model_file)
model.load_state_dict(torch.load(model_file)['state_dict'], strict=True)
for m in model.modules():
    m.eval()
model.eval()
model.cuda()


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


def get_imlist():
    data_dir = '/private/ningqingqun/bags/crane/howo1_2019_11_28_12_00_08_0.msg/front_left/201911281200'
    data_dir = '/private/ningqingqun/bags/crane/howo1_2019_12_04_09_14_54_0.msg/front_left/201912040915'
    data_dir = '/private/ningqingqun/bags/crane/howo1_2019_12_07_14_24_11_19.msg/front_left/201912071424'
    out_dir = '/private/ningqingqun/results/centernet_results/' + data_dir.split(
        '/')[-1] + 'front_left'
    im_list = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('.jpg')
    ]
    return im_list, out_dir


def get_imlist2():
    split = 'train190902'
    data_root = '/private/ningqingqun/obstacle_detector/data/obstacle2d'
    list_file = os.path.join(data_root, 'ImageSets', f'{split}.txt')
    im_prefix = os.path.join(data_root, 'JPGImages')
    with open(list_file) as f:
        imlist = f.read().splitlines()
    imlist = [os.path.join(im_prefix, l + '.jpg') for l in imlist]

    out_dir = f'/private/ningqingqun/results/centernet_results/{split}_txt'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    print('results will save to {}'.format(out_dir))

    return imlist, out_dir


def verify_images():
    im_list, out_dir = get_imlist2()

    detections = {}
    for imf in tqdm(im_list, total=len(im_list)):
        im = cv2.imread(imf)
        reiszed_im = cv2.resize(im, (1280, 800))
        inputs = preprocess(reiszed_im)

        result = model(inputs.cuda(), return_loss=False, try_dummy=False)
        labels = [i * np.ones((len(r), 1)) for i, r in enumerate(result, 1)]
        bboxes = np.vstack(result)
        labels = np.vstack(labels)

        # nms for all classes
        bboxes, idx = nms(bboxes, 0.7)
        labels = labels[idx]

        # detection above score threshold
        idx = bboxes[:, -1] > thresh
        bboxes = bboxes[idx]
        labels = labels[idx]
        detections[imf] = np.hstack([bboxes, labels])
        continue

        out_file = os.path.join(out_dir, os.path.basename(imf))
        show_result(
            reiszed_im,
            result,
            class_names,
            score_thr=thresh,
            out_file=out_file)
    out_file = os.path.join(out_dir, 'detections.pkl')
    with open(out_file, 'wb') as f:
        pkl.dump(detections, f)


def verify_image():
    imf = '/private/ningqingqun/bags/crane/howo1_2019_11_05_09_06_22_6.msg/front_right/201911050907/201911050907_00000417_1572916023857.jpg'
    im = cv2.imread(imf)
    reiszed_im = cv2.resize(im, (1280, 800))
    inputs = preprocess(reiszed_im)

    result = model(inputs.cuda(), return_loss=False, try_dummy=False)
    show_result(
        reiszed_im,
        result,
        class_names,
        score_thr=thresh,
        out_file='centernet_verify.jpg')


def verify_video():
    out_dir = '/private/ningqingqun/results/centernet_results/senyun'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    vfile = '/private/ningqingqun/datasets/videos/WIN_20191030_20_38_42_Pro.mp4'
    cap = cv2.VideoCapture(vfile)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(total_frame)
    display_num = 20
    frame_count = 0
    base_frame_index = random.randint(0, total_frame - display_num)
    cap.set(cv2.CAP_PROP_POS_FRAMES, base_frame_index)
    print('process from index: {}'.format(base_frame_index))
    while (cap.isOpened()):
        ret, frame = cap.read()
        reiszed_im = cv2.resize(frame, (1280, 800))
        inputs = preprocess(reiszed_im)

        result = model(inputs.cuda(), return_loss=False, try_dummy=False)
        out_file = os.path.join(
            out_dir, '{:04}.jpg'.format(base_frame_index + frame_count))
        show_result(
            reiszed_im,
            result,
            class_names,
            score_thr=thresh,
            out_file=out_file)

        frame_count += 1
        if frame_count > display_num:
            break

    cap.release()


thresh = 0.4
class_names = zoo[task]['classes']
verify_images()