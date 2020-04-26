import numpy as np
from .utils import to_tensor
from mmcv.parallel import DataContainer as DC

import torch
from torch.utils.data import Dataset


class GTMixupDetection(Dataset):
    """Detection dataset wrapper that performs gt flip mixup for normal dataset.

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
        elif self._mixup_args is None:
            self._mixup_args = (0.5, 0.5)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        data = self._dataset[idx]
        lambd = 1

        # draw a random lambda ratio from distribution
        if self._mixup is not None:
            lambd = max(0, min(1, self._mixup(*self._mixup_args)))

        if lambd >= 1:
            return data

        lambd = 1 - lambd if lambd < 0.5 else lambd
        img, boxes, label, meta = data['img'].data, data['gt_bboxes'].data, \
                                  data['gt_labels'].data, data['img_meta'].data

        int_boxes = boxes.int()
        for box in int_boxes:
            img[:, box[1]:box[3], box[0]:box[2]] = \
                img[:, box[1]:box[3], box[0]:box[2]] * lambd + \
                torch.flip(img[:, box[1]:box[3], box[0]:box[2]], dims=(2,)) * (1 - lambd)

        mix_img = DC(to_tensor(img), stack=True)
        mix_boxes = DC(to_tensor((boxes.numpy())))
        mix_labels = DC(to_tensor(label.numpy()))
        mix_meta = DC(meta, cpu_only=True)
        mix_data = dict(img=mix_img, img_meta=mix_meta, gt_bboxes=mix_boxes, gt_labels=mix_labels)
        return mix_data
