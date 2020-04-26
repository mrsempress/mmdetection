import mmcv
import numpy as np
from numpy import random

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels):
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, boxes, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img

        # support both bbox and corners
        # bbox shape: (n,4)
        # corners shape: (n, 6, 2)
        if boxes.ndim == 2 and boxes.shape[1] == 4:
            boxes += np.tile((left, top), 2)
        elif boxes.ndim == 3 and boxes.shape[2] == 2:
            boxes += np.array((left, top))
        return img, boxes, labels


class RandomCrop(object):

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, labels):
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, boxes, labels

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                return img, boxes, labels


class RandomCornersCrop(object):

    def __init__(self, min_crop_sizes=(0.3, 0.5, 0.7, 0.9, 1)):
        # 1: return ori img
        self.min_crop_sizes = min_crop_sizes

    def __call__(self, img, corners, labels):
        h, w, c = img.shape
        hw_ratio = float(h) / w
        left_pose = [0, 2, 2, 4, 4, 6, 6, 0]
        right_pose = [0, 0, 2, 2, 4, 4, 6, 6]

        min_crop_size = random.choice(self.min_crop_sizes)
        if min_crop_size == 1:
            return img, corners, labels

        for i in range(50):
            new_w = random.uniform(min_crop_size * w, w)
            new_h = random.uniform(min_crop_size * h, h)

            # h / w in [0.5, 2]*hw_ratio
            if new_h / new_w < 0.5 * hw_ratio or new_h / new_w > 2 * hw_ratio:
                continue

            left = int(random.uniform(w - new_w))
            top = int(random.uniform(h - new_h))
            right = int(left + new_w)
            bottom = int(top + new_h)

            # at least one object inside the crop img
            center = corners[:, [0, 2, 3, 5], :].mean(axis=1)
            center_mask = (center[:, 0] > left) * (center[:, 1] > top) * (
                center[:, 0] < right) * (
                    center[:, 1] < bottom)
            if not center_mask.any():
                continue

            # 1. filter objects  outside crop img
            valid_mask = (corners[..., 0] > left) * (
                corners[..., 0] < right) * (corners[..., 1] > top) * (
                    corners[..., 1] < bottom)
            valid_mask = valid_mask.sum(axis=1) > 0
            corners = corners[valid_mask]
            labels = labels[valid_mask]

            # 2. update pose changed by cropping
            pose = labels[:, 1]
            is_bi_side = pose % 2 == 1
            update_mask = (
                (corners[..., 0] > left).sum(axis=1) <= 2) * is_bi_side
            pose[update_mask] = (pose[update_mask] + 7) % 8
            update_mask = (
                (corners[..., 0] < right).sum(axis=1) <= 2) * is_bi_side
            pose[update_mask] = (pose[update_mask] + 1) % 8
            labels[:, 1] = pose

            # adjust boxes
            img = img[top:bottom, left:right]
            corners -= np.array((left, top))
            break

        return img, corners, labels


class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None,
                 random_corner_crop=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))
        if random_corner_crop is not None:
            self.transforms.append(RandomCornersCrop(**random_corner_crop))

    def __call__(self, img, boxes, labels):
        img = img.astype(np.float32)
        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)
        return img, boxes, labels
