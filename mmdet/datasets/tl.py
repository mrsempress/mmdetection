import os
import numpy as np

from mmdet.core.utils import logger
from .custom import CustomDataset


class TLDataset(CustomDataset):
    CLASSES = (0,
               11, 12, 13, 14, 15, 16, 17, 18,
               21, 22, 23, 24, 25, 26, 27, 28,
               31)
    IGNORE_CLASSES = (6,)

    def __init__(self,
                 *args,
                 with_mask=False,
                 with_crowd=False,
                 **kwargs):
        assert with_mask == False, "Not support mask yet."
        assert with_crowd == False, "Not support crowd yet."
        self.class_to_ind = dict(zip(self.CLASSES + self.IGNORE_CLASSES,
                                     range(len(self.CLASSES + self.IGNORE_CLASSES))))
        # load_annotations is called before this statement in CustomDataset
        self.with_crowd = with_crowd
        super(TLDataset, self).__init__(*args, with_mask=with_mask, with_crowd=with_crowd, **kwargs)

    def load_annotations(self, ann_file):
        imglist_file = ann_file
        ann_dir = os.path.join(*ann_file.split('/')[:-1])
        with open(imglist_file, 'r') as f:
            lines = f.readlines()
        self.imglist = [line.strip() for line in lines]
        img_infos = []
        for img_name in self.imglist:
            img_dict = dict()
            img_dict['filename'] = img_name + '.jpg'
            is_img_valid_flag = self._add_detection_gt(ann_dir, img_name, img_dict)
            if is_img_valid_flag:
                img_infos.append(img_dict)
        return img_infos

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def _add_detection_gt(self, ann_dir, img_name, img_dict):
        anno_file = os.path.join(ann_dir, 'Annotations', img_name + '.txt')
        # The assertion below may slow down the speed dramatically.
        # assert os.path.isfile(anno_file), anno_file
        with open(anno_file, 'r') as f:
            lines = f.readlines()

        # clean-up boxes
        valid_bboxes = []
        ignore_info = []
        for line in lines:
            line = line.strip().split()
            assert len(line) == 7, 'data format not suport {}'.format(line)

            line = list(map(int, line))
            name = line[0]
            cover = line[2]
            ignore = line[1] if cover == 0 else 1
            bbox = line[3:]

            if name in self.IGNORE_CLASSES:
                if not self.with_crowd or ignore == 0:
                    continue
                else:
                    ignore = 1

            # hard code, will be removed in the future
            name = 31 if name == 3 else name
            name = (10 + name % 10) if name // 10 == 3 and name != 31 else name

            if name not in self.CLASSES and name not in self.IGNORE_CLASSES:
                logger.warn("Unknow class type {} found in {}, the box will be removed".format(
                    name, img_dict['filename']))
                continue

            # bbox is originally in float
            # x1/y1 means upper-left corner and w/h means true w/h. This can be verified by segmentation pixels.
            # But we do make an assumption here that (0.0, 0.0) is upper-left corner of the first pixel
            bbox = list(map(float, bbox))

            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                logger.warn("Incorrect box shape in {}, remove the img".format(
                    img_dict['filename']))
                return False

            assert (len(bbox + [name]) == 5)
            valid_bboxes.append(bbox + [name])
            ignore_info.append(ignore)

        # all geometrically-valid boxes are returned
        boxes = np.asarray([bbox[:-1] for bbox in valid_bboxes], dtype='float32')  # (n, 4)
        cls = np.asarray([self.class_to_ind[bbox[-1]] for bbox in valid_bboxes],
                         dtype='int64')  # (n,)
        is_crowd = np.asarray(ignore_info, dtype='int8')

        ignore_boxes = None
        if self.with_crowd and is_crowd.any():
            ignore_boxes = boxes[is_crowd == 0]
            boxes = boxes[is_crowd == 1]
            cls = cls[is_crowd == 1]

        # add the keys
        img_dict['width'] = 1280
        img_dict['height'] = 640
        img_dict['ann'] = {'bboxes': boxes, 'labels': cls, 'bboxes_ignore': ignore_boxes}
        return True
