import os
import numpy as np
import pickle

from mmdet.core.utils import logger
from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class Vision2DDataset(CustomDataset):
    CLASSES = ('car', 'van', 'bus', 'truck', 'forklift',
               'person', 'person-sitting',
               'bicycle', 'motor', 'open-tricycle', 'close-tricycle',
               'water-block', 'cone-block', 'other-block', 'crash-block', 'triangle-block',
               'warning-block', 'small-block', 'large-block')
    IGNORE_CLASSES = ('bicycle-group', 'person-group', 'motor-group',
                      'parked-bicycle', 'parked-motor', 'cross-bar')
    CLASSES_MAP = {
        '8cls': {
            'classes': ('__background__',  # always index 0
                        'car', 'bus', 'truck', 'person', 'bicycle', 'tricycle', 'block'),
            'map': {
                'car': ['car', 'van'],
                'bus': ['bus'],
                'truck': ['truck', 'forklift'],
                'person': ['person', 'person-sitting'],
                'bicycle': ['bicycle', 'motor'],
                'tricycle': ['open-tricycle', 'close-tricycle', 'tricycle'],
                'block': ['water-block', 'cone-block', 'other-block', 'crash-block',
                          'triangle-block', 'warning-block',
                          'small-block', 'large-block', 'block'],
            },
            'ignore_classes': IGNORE_CLASSES,
        }
    }

    def __init__(self,
                 ann_file,
                 pipeline,
                 keep_difficult=False,
                 task='8cls',
                 **kwargs):
        self.classes = Vision2DDataset.CLASSES_MAP[task]['classes']
        self.id_map = self._convert_map(Vision2DDataset.CLASSES_MAP[task]['map'])
        id_map_rev = {}
        for k, v in self.id_map.items():
            id_map_rev[v] = k
        id_map_rev_key_sorted = sorted(list(id_map_rev.keys()))
        self.cat_ids = []
        for k in id_map_rev_key_sorted:
            self.cat_ids.append(id_map_rev[k])
        self.keep_difficult = keep_difficult
        # load_annotations is called before this statement in CustomDataset
        super(Vision2DDataset, self).__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_file):
        imglist_file = ann_file
        start_with_root = True if ann_file[0] == '/' else False
        ann_dir = os.path.join(*ann_file.split('/')[:-2])
        if start_with_root:
            ann_dir = '/' + ann_dir
        with open(imglist_file, 'r') as f:
            self.img_ids = f.read().splitlines()

        pkl_dir = os.path.join('data', 'pkl')
        if not os.path.isdir(pkl_dir):
            os.makedirs(pkl_dir)
        pkl_path = os.path.join(pkl_dir, os.path.basename(ann_file).split('.')[0] + '.pkl')

        if os.path.isfile(pkl_path):
            img_infos = pickle.load(open(pkl_path, "rb"))
        else:
            img_infos = []
            for img_name in self.img_ids:
                img_dict = dict()
                if '/2009' in img_name:
                    img_name.replace('/camera', '')
                img_dict['filename'] = img_name + '.jpg'
                is_img_valid_flag = self._add_detection_gt(ann_dir, img_name, img_dict)
                if is_img_valid_flag:
                    img_infos.append(img_dict)
            pickle.dump(img_infos, open(pkl_path, 'wb'))
        return img_infos

    def _add_detection_gt(self, ann_dir, img_name, img_dict):
        anno_file = os.path.join(ann_dir, 'Annotations', img_name + '.txt')
        # The assertion below may slow down the speed dramatically.
        # assert os.path.isfile(anno_file), anno_file
        with open(anno_file, 'r') as f:
            annotations = f.read().splitlines()

        # clean-up boxes
        valid_bboxes = []
        ignore_info = []
        difficult_mark = []
        for line in annotations:
            content = line.split(' ')
            if len(content) == 6:
                x1, y1, x2, y2, occ, categ = content
                difficult = int(occ) == 2
            elif len(content) == 7:
                x1, y1, x2, y2, occ, categ, difficult = content
                difficult = (int(occ) == 2 or int(difficult) == 1)
            else:
                logger.warn("Unknow content {} found in {}, remove".format(
                    content, img_dict['filename']))

            if categ not in self.CLASSES and categ not in self.IGNORE_CLASSES:
                logger.warn("Unknow class type {} found in {}, the box will be removed".format(
                    categ, img_dict['filename']))
                continue

            if categ in Vision2DDataset.IGNORE_CLASSES:
                continue

            if not self.keep_difficult and difficult:
                continue

            difficult_mark.append(difficult)
            bbox = list(map(float, [x1, y1, x2, y2]))
            valid_bboxes.append(bbox + [self.id_map[categ]])
            ignore_info.append(1)

        if len(valid_bboxes) == 0:
            return False

        # all geometrically-valid boxes are returned
        boxes = np.asarray([bbox[:-1] for bbox in valid_bboxes], dtype='float32')  # (n, 4)
        cls = np.asarray([bbox[-1] for bbox in valid_bboxes],
                         dtype='int64')  # (n,)
        is_crowd = np.asarray(ignore_info, dtype='int8')

        ignore_boxes = None
        # if self.with_crowd and is_crowd.any():
        #     ignore_boxes = boxes[is_crowd == 0]
        #     boxes = boxes[is_crowd == 1]
        #     cls = cls[is_crowd == 1]

        if ignore_boxes:
            ignore_boxes = np.array(ignore_boxes, dtype=np.float32)
        else:
            ignore_boxes = np.zeros((0, 4), dtype=np.float32)

        # add the keys
        img_dict['width'] = 1920
        img_dict['height'] = 1200
        img_dict['ann'] = {'bboxes': boxes, 'labels': cls}
        return True

    def _convert_map(self, m):
        converted = {}
        for key in m:
            main_id = self.classes.index(key)
            for item in m[key]:
                assert item not in converted, 'Conversion Error'
                converted[item] = main_id
        return converted
