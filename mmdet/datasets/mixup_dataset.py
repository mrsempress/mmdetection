import numpy as np
from .utils import to_tensor
from mmcv.parallel import DataContainer as DC

from torch.utils.data import Dataset


class MixupDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset.

    Parameters
    ----------
    dataset : CustomDataset.
    mixup : callable random generator, e.g. np.random.uniform
        A random mixup ratio sampler, preferably a random generator from numpy.random
        A random float will be sampled each time with mixup(*args).
        Use None to disable.
    *args : list
        Additional arguments for mixup random sampler.

    """
    def __init__(self, dataset, mixup=None, debug=False, mixup_args=None):
        self._dataset = dataset
        self.CLASSES = dataset.CLASSES
        self._mixup = mixup
        self._mixup_args = mixup_args
        self._debug = debug
        self.flag = self._dataset.flag if hasattr(self._dataset, 'flag') else None

    def set_mixup(self, mixup=None, *args):
        """Set mixup random sampler, use None to disable.

        Parameters
        ----------
        mixup : callable random generator, e.g. np.random.uniform
            A random mixup ratio sampler, preferably a random generator from numpy.random
            A random float will be sampled each time with mixup(*args)
        *args : list
            Additional arguments for mixup random sampler.

        """
        self._mixup = mixup
        if args:
            self._mixup_args = args
        elif self._mixup_args == None:
            self._mixup_args = (0.5, 0.5)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        # first image
        data1 = self._dataset[idx]
        lambd = 1

        # draw a random lambda ratio from distribution
        if self._mixup is not None:
            lambd = max(0, min(1, self._mixup(*self._mixup_args)))

        if lambd >= 1:
            # weights1 = np.ones((label1.shape[0], 1))
            # label1 = np.hstack((label1, weights1))
            return data1

        img1, boxes1, label1 , meta1 = data1['img'].data, data1['gt_bboxes'].data, \
                                       data1['gt_labels'].data, data1['img_meta'].data
        # second image
        idx2 = np.random.choice(np.delete(np.arange(len(self)), idx))
        data2 = self._dataset[idx2]
        img2, boxes2, label2, meta2 = data2['img'].data, data2['gt_bboxes'].data, \
                                      data2['gt_labels'].data, data2['img_meta'].data
        # mixup two images
        pad_height = max(meta1['pad_shape'][0], meta2['pad_shape'][0])
        pad_width = max(meta1['pad_shape'][1], meta2['pad_shape'][1])
        img_height = max(meta1['img_shape'][0], meta2['img_shape'][0])
        img_width = max(meta1['img_shape'][1], meta2['img_shape'][1])

        mix_img = np.zeros(shape=(3, pad_height, pad_width), dtype='float32')

        # label1 = np.hstack((label1, np.full((label1.shape[0], 1), lambd)))
        # label2 = np.hstack((label2, np.full((label2.shape[0], 1), 1. - lambd)))

        pad_shape = (pad_height, pad_width, 3)
        img_shape = (img_height, img_width, 3)
        mixup_params = dict(lambd=lambd, first_n_labels=len(label1))
        mix_meta = DC(dict(ori_shape=None, img_shape=img_shape, pad_shape=pad_shape,
                           scale_factor=None, flip=None, mixup_params=mixup_params), cpu_only=True)

        if self._debug:
            mix_img[:, :meta1['pad_shape'][0], :meta1['pad_shape'][1]] = \
                img1.float().numpy()
            mix_img = DC(to_tensor(mix_img), stack=True)
            mix_boxes = DC(to_tensor(boxes1.numpy()))
            mix_labels = DC(to_tensor(label1.numpy()))
            mix_data = dict(img=mix_img, img_meta=mix_meta, gt_bboxes=mix_boxes,
                            gt_labels=mix_labels)
            return mix_data

        mix_img[:, :meta1['pad_shape'][0], :meta1['pad_shape'][1]] = \
            img1.float().numpy() * lambd
        mix_img[:, :meta2['pad_shape'][0], :meta2['pad_shape'][1]] += \
            img2.float().numpy() * (1. - lambd)

        mix_img = DC(to_tensor(mix_img), stack=True)
        mix_boxes = DC(to_tensor(np.vstack((boxes1.numpy(), boxes2.numpy()))))
        mix_labels = DC(to_tensor(np.concatenate((label1.numpy(), label2.numpy()))))
        mix_data = dict(img=mix_img, img_meta=mix_meta, gt_bboxes=mix_boxes, gt_labels=mix_labels)
        return mix_data
