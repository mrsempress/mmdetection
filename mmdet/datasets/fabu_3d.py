import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .registry import DATASETS
from .transforms import (ImageTransform, PointTransform, Numpy2Tensor)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation


@DATASETS.register_module
class Vision3DDataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'url': 'http://dataplatform.fabu.ai/xxxx',
            'width': 1920,
            'height': 1200,
            'ann': {
                'corners': <np.ndarray> (n, 6, 2),
                'classes': <np.ndarray> (n),
                'poses': <np.ndarray> (n),
                'corners_ignore': <np.ndarray> (n, 6, 2),
                'bboxes': <np.ndarray> (n,4),
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = [
        '__background__', 'car', 'bus', 'truck', 'person', 'bicycle',
        'tricycle', 'block'
    ]

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 flip_ratio=0,
                 with_ignore=False,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 sample_file=None,
                 num_classes=8,
                 test_mode=False):
        super(Vision3DDataset, self).__init__()
        # prefix of images path
        self.img_prefix = img_prefix
        self.name = osp.basename(ann_file).split('.')[0]

        # load annotations (and proposals)
        self.raw_annotations = self.load_annotations(ann_file)

        # support dict or list
        if isinstance(self.raw_annotations, list):
            self.ids = range(len(self.raw_annotations))
        elif isinstance(self.raw_annotations, dict):
            if sample_file is not None and osp.isfile(sample_file):
                self.ids = mmcv.load(sample_file, encoding='latin1')
            else:
                self.ids = sorted(list(self.raw_annotations.keys()))
        else:
            raise Exception("Unrecognized type of annotations: {}".format(
                type(self.raw_annotations)))

        # filter images with no annotation during training
        if not test_mode:
            self._filter_imgs()

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # some datasets provide bbox annotations as ignore/crowd/difficult,
        self.with_ignore = with_ignore

        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.point_transform = PointTransform()
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

        self.num_classes = num_classes

    def __len__(self):
        return len(self.ids)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file, encoding='latin1')

    def get_ann_info(self, idx):
        key = self.ids[idx]
        return self.raw_annotations[key]['ann']

    def get_img_info(self, idx):
        key = self.ids[idx]
        return self.raw_annotations[key]

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for idx in self.ids:
            img_info = self.raw_annotations[idx]
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(idx)
        self.ids = valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.get_img_info(i)
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):

        img_info = self.get_img_info(idx)
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))

        ann = self.get_ann_info(idx)
        gt_corners = ann['corners']
        gt_classes = ann['classes']
        gt_poses = ann['poses']

        # class-id with negtive number are encoded as ignored
        if not self.with_ignore:
            idx = np.where(gt_classes > 0)[0]
            gt_corners = gt_corners[idx]
            gt_classes = gt_classes[idx]
            gt_poses = gt_poses[idx]

        if len(gt_corners) == 0:
            return None
        ## ensure the corner order.
        ## bi-sideview:
        ##                      c0 -- c1 -- c2
        ##                       |     |     |
        ##                      c3 -- c4 -- c5
        ## single-sideview
        ##                      c0 --  - -- c2
        ##                       |     |     |
        ##                      c3 --  - -- c5
        try:
            assert (gt_corners[:, 0, 0] < gt_corners[:, 2, 0]).all()
        except:
            print(gt_corners.shape)
        assert (gt_corners[:, 0, 1] < gt_corners[:, 3, 1]).all()
        assert (gt_corners[:, 5, 0] > gt_corners[:, 3, 0]).all()
        assert (gt_corners[:, 5, 1] > gt_corners[:, 2, 1]).all()

        # extra augmentation
        if self.extra_aug is not None:
            labels = np.stack([gt_classes, gt_poses], axis=1)
            img, gt_corners, labels = self.extra_aug(img, gt_corners, labels)
            gt_classes, gt_poses = np.split(labels, 2, axis=1)
            gt_classes = gt_classes.reshape(-1)
            gt_poses = gt_poses.reshape(-1)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()

        gt_corners, gt_poses = self.point_transform(gt_corners, gt_poses,
                                                    img_shape,
                                                    scale_factor[:2], flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        # vimg = img * np.array(
        #     self.img_norm_cfg['mean'],
        #     dtype=np.float32).reshape(3, 1, 1) + np.array(
        #         self.img_norm_cfg['std'], dtype=np.float32).reshape(3, 1, 1)
        # show_train_img(vimg.astype(np.uint8), gt_corners, gt_poses)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_corners=DC(to_tensor(gt_corners)),
            gt_classes=DC(to_tensor(gt_classes)),
            gt_poses=DC(to_tensor(gt_poses)),
        )

        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.get_img_info(idx)
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))

        def prepare_single(img, scale, flip):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            return _img, _img_meta

        imgs = []
        img_metas = []
        for scale in self.img_scales:
            _img, _img_meta = prepare_single(img, scale, False)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            if self.flip_ratio > 0:
                _img, _img_meta = prepare_single(img, scale, True)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
        data = dict(img=imgs, img_meta=img_metas)
        return data


def show_train_img(img, gt_corners, gt_poses):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    pose2sides = {
        0: [('tail', 'outer')],
        1: [('left', 'left'), ('tail', 'right')],
        2: [('left', 'outer')],
        3: [('head', 'left'), ('left', 'right')],
        4: [('head', 'outer')],
        5: [('right', 'left'), ('head', 'right')],
        6: [('right', 'outer')],
        7: [('tail', 'left'), ('right', 'right')],
    }
    side2color = {
        'left': 'yellow',
        'head': 'red',
        'right': 'gold',
        'tail': 'green',
    }
    side2idx = {
        'left': [0, 1, 4, 3],
        'right': [1, 2, 5, 4],
        'outer': [0, 2, 5, 3],
    }

    plt.figure(figsize=(30, 48))
    ax = plt.gca()
    for i in range(len(gt_poses)):
        if gt_poses[i] < 0:
            side_list = [('tail', 'outer')]
        else:
            side_list = pose2sides[gt_poses[i]]
        for s, v in side_list:
            color = side2color[s]
            vcorners = gt_corners[i, side2idx[v], :]
            ax.add_patch((Polygon(vcorners, alpha=0.5, facecolor=color)))

    if img.shape[2] == 3:
        plt.imshow(img[..., [2, 1, 0]])
    else:
        plt.imshow(img.transpose([1, 2, 0]))
    plt.show()
