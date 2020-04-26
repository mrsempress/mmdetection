import numpy as np
import mmcv
from collections import defaultdict
from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class CraneMarkDataset(CustomDataset):
    RAW_CLASSES = [
        'right_first_number', 'right_second_number', 'right_third_number',
        'right_first_line', 'right_second_line', 'right_third_line',
        'left_first_number', 'left_second_number', 'left_third_number',
        'left_first_line', 'left_second_line', 'left_third_line', 'NO31',
        'NO32', 'NO33', 'NO34', 'NO35', 'NO36'
    ]
    CLASSES = [
        'right20',
        'right40',
        'right45',
        'left20',
        'left40',
        'left45',
        'NO31',
        'NO32',
        'NO33',
        'NO34',
        'NO35',
        'NO36',
    ]
    CLASS_MAP = {
        'right20': ['right_first_line', 'right_first_number'],
        'right40': ['right_second_line', 'right_second_number'],
        'right45': ['right_third_line', 'right_third_number'],
        'left20': ['left_first_line', 'left_first_number'],
        'left40': ['left_second_line', 'left_second_number'],
        'left45': ['left_third_line', 'left_third_number'],
    }
    CLASS_MAP_INV = {
        'right_first_line': 'right20',
        'right_second_line': 'right40',
        'right_third_line': 'right45',
        'left_first_line': 'left20',
        'left_second_line': 'left40',
        'left_third_line': 'left45',
    }

    def load_annotations(self, ann_file):
        infos = mmcv.load(ann_file)

        infos_list = []
        for key, ann in infos.items():
            infos_list.append(self.generate_bbox(ann))

        return infos_list

    def generate_bbox(self, annotations):
        raw_corners = annotations['ann']['corners']
        raw_classes = annotations['ann']['classes']

        number_list = defaultdict(list)
        line_list = []
        bboxes = []
        labels = []
        for i in range(len(raw_corners)):
            cls_id = raw_classes[i]
            cls_name = self.RAW_CLASSES[cls_id]
            if 'number' in cls_name:
                number_list[cls_name].append(i)
            elif 'line' in cls_name:
                line_list.append(i)
            else:
                labels.append(self.CLASSES.index(cls_name) + 1)
                bboxes.append(corners2bboxes(raw_corners[i]))

        for i in line_list:
            line_name = self.RAW_CLASSES[raw_classes[i]]
            number_name = line_name.replace('line', 'number')
            line_corners = raw_corners[i]

            matched_number_list = number_list[line_name]
            if matched_number_list:
                d_min = 1e4
                for ni in matched_number_list:
                    d = np.abs(line_corners[0, 0] - raw_corners[ni, 0, 0])
                    if d < d_min:
                        d_min = d
                        num_corners = raw_corners[ni]
                bboxes.append(
                    corners2bboxes(np.vstack([line_corners, num_corners])))
            else:
                bboxes.append(corners2bboxes(line_corners))
            labels.append(
                self.CLASSES.index(self.CLASS_MAP_INV[line_name]) + 1)

        ann = dict(
            bboxes=np.array(bboxes, dtype=np.float32),
            labels=np.array(labels, dtype=np.int32),
        )
        annotations['ann'] = ann
        return annotations


def corners2bboxes(corners):
    x1 = corners[:, 0].min()
    x2 = corners[:, 0].max()
    y1 = corners[:, 1].min()
    y2 = corners[:, 1].max()
    bboxes = np.array([x1, y1, x2, y2], dtype=np.float32)

    return bboxes