import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import multi_apply, multiclass_nms
from mmdet.core.bbox import bbox_overlaps, center_to_point, point_to_center
from mmdet.models.losses import weighted_l1
from .anchor_head import AnchorHead
from ..registry import HEADS


class YOLOv3Output(nn.Module):

    def __init__(self,
                 in_channel,
                 anchors,
                 stride,
                 num_classes,
                 alloc_size=(128, 128)):
        super(YOLOv3Output, self).__init__()
        anchors = np.array(anchors).astype('float32')
        self.num_classes = num_classes
        self.num_pred = 1 + 4 + self.num_classes  # 1 objness + 4 box + num_class
        self.num_anchors = anchors.size // 2  # 3 for a single level.
        self.stride = stride

        self.prediction = nn.Conv2d(in_channel, self.num_pred * self.num_anchors,
                                    kernel_size=1, padding=0, stride=1)

        # anchors will be multiplied to predictions, shape: (1, 1, 3, 2).
        # **the anchors are just used to provide the base scale for each location.**
        self.anchors = torch.tensor(anchors.reshape(1, 1, -1, 2), dtype=torch.float32).cuda()
        # offsets will be added to predictions
        grid_x = np.arange(alloc_size[1])
        grid_y = np.arange(alloc_size[0])
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        # concat to (n, n, 2) and expand dims to (1, 1, n, n, 2). so it's easier for broadcasting.
        # **the offsets are used to represent the top-left corner of each grid in origin scale.**
        offsets = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)
        self.offsets = np.expand_dims(np.expand_dims(offsets, axis=0), axis=0)

    def forward(self, x):
        # prediction flat to (batch, pred per pixel, height * width)
        pred = self.prediction(x).view((x.size(0), self.num_anchors * self.num_pred, -1))
        # transpose to (batch, height * width, num_anchors, num_pred)
        pred = pred.transpose(1, 2).view((x.size(0), -1, self.num_anchors, self.num_pred))

        # components. different anchors at same location correspond to different prediction.
        raw_box_centers = pred[:, :, :, :2]  # (batch, height * width, num_anchors, 2)
        raw_box_scales = pred[:, :, :, 2:4]
        objness = pred[:, :, :, 4:5]
        class_pred = pred[:, :, :, 5:]  # 80 in coco, exclude background.

        # valid offsets, (1, 1, height, width, 2)
        offsets = raw_box_centers.new_tensor(self.offsets.copy()[0, 0, :x.size(2), :x.size(3), :])
        # reshape to (1, height * width, 1, 2)
        offsets = offsets.view((1, -1, 1, 2))

        # note that box_centers are limited in the grid, and different anchors are corresponding
        # to different box centers, scales, confidences and scores.
        box_centers = (torch.sigmoid(raw_box_centers) + offsets) * self.stride
        box_scales = raw_box_scales.exp() * self.anchors
        confidence = torch.sigmoid(objness)
        class_score = torch.sigmoid(class_pred) * confidence
        wh = box_scales / 2.0
        bboxes = torch.cat((box_centers - wh, box_centers + wh), dim=-1)
        # bboxes: (batch, h*w*num_anchors, 4).
        bboxes = bboxes.view(bboxes.size(0), -1, 4)

        if self.training:
            return (bboxes, raw_box_centers, raw_box_scales, objness, class_pred,
                    self.anchors, offsets)

        # scores: (batch, h*w*num_anchors, 80).
        scores = class_score.view(class_score.size(0), -1, self.num_classes)
        return scores, bboxes


@HEADS.register_module
class YOLOv3Head(AnchorHead):

    def __init__(self,
                 in_channels=(256, 512, 1024),
                 anchors=((10, 13, 16, 30, 33, 23),
                          (30, 61, 62, 45, 59, 119),
                          (116, 90, 156, 198, 373, 326)),
                 strides=(8, 16, 32),
                 num_classes=81,
                 alloc_size=(128, 128),
                 ignore_iou_thresh=0.7):
        super(AnchorHead, self).__init__()
        self.num_classes = num_classes - 1  # bg is replaced by the objness in yolov3 head.

        outputs = []
        # the output layers are used in reverse order.
        for i, (channel, anchor, stride) in enumerate(zip(
                in_channels[::-1], anchors[::-1], strides[::-1])):
            outputs.append(YOLOv3Output(channel, anchor, stride,
                                        num_classes=self.num_classes, alloc_size=alloc_size))
        self.yolo_outputs = nn.Sequential(*outputs)

        self._target_generator = YOLOV3TargetGenerator(self.num_classes, ignore_iou_thresh)
        self._loss = YOLOV3Loss()

    def init_weights(self):
        for l in self.yolo_outputs.modules():
            if isinstance(l, nn.Conv2d):
                normal_init(l, std=0.01)

    def forward(self, feats):
        """
        YOLOv3: D53 takes 18.1ms, HEAD takes 10.3ms.

        | YOLOv3 | block  | output | transtition+upsample | Total |
        | ------ | ------ | ------ | -------------------- | ----- |
        | lv5    | 2.76ms | 2.61ms | 2.28ms               | 4.1ms |
        | lv4    | 0.88ms | 0.56ms | 0.38ms               | 3.7ms |
        | lv3    | 0.52ms | 0.94ms | NA                   | 2.6ms |

        Args:
            feats: list(tensor) <=> [lv3, lv4, lv5].

        Returns:

        """
        all_scores = []
        all_bboxes = []
        all_box_centers = []
        all_box_scales = []
        all_objectness = []
        all_class_pred = []
        all_anchors = []
        all_offsets = []
        all_feat_maps = []
        # the YOLO output layers are used in reverse order, i.e., from very deep layers to shallow.
        for i, (x, output) in enumerate(zip(feats[::-1], self.yolo_outputs)):
            if self.training:
                bboxes, box_centers, box_scales, objness, class_pred, anchors, offsets = output(x)
                # (batch, h*w, num_anchors, xx) => (batch, h*w*num_anchors, xx).
                all_box_centers.append(
                    box_centers.contiguous().view((box_centers.size(0), -1, box_centers.size(-1))))
                all_box_scales.append(
                    box_scales.contiguous().view((box_scales.size(0), -1, box_scales.size(-1))))
                all_class_pred.append(
                    class_pred.contiguous().view((class_pred.size(0), -1, class_pred.size(-1))))
                all_objectness.append(
                    objness.contiguous().view((objness.size(0), -1, objness.size(-1))))
                all_anchors.append(anchors)
                all_offsets.append(offsets)
                # here we use fake featmap to reduce memory consuption, only shape[2, 3] is used
                fake_featmap = torch.zeros_like(x[0, 0, :, :])
                all_feat_maps.append(fake_featmap.view(1, 1, *fake_featmap.shape))
            else:
                scores, bboxes = output(x)
                all_scores.append(scores)

            all_bboxes.append(bboxes)

        if self.training:
            return (all_bboxes, all_box_centers, all_box_scales, all_class_pred,
                    all_objectness, all_anchors, all_offsets, all_feat_maps)

        return all_scores, all_bboxes

    def get_bboxes(self, scores, bboxes, img_metas, cfg, rescale=False):
        """Note that the bbox_preds in yolov3 is not delta, but the decoded box.

        Args:
            scores: list(tensor), tensor <=> (batch, h * w * num_anchor, 80). the confidence is
                included in the scores.
            bboxes: list(tensor), tensor <=> (batch, h * w * num_anchor, 4).
            img_metas: list(dict).
            cfg: test cfg.
            rescale: if to apply the scale factor in the img_metas to boxes.

        Returns:
            list(tuple).
        """
        assert len(scores) == len(bboxes)
        num_levels = len(scores)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bboxes[i][img_id].detach() for i in range(num_levels)
            ]
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single_image(
                cls_score_list, bbox_pred_list, cfg, scale_factor, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single_image(self,
                                cls_scores,
                                bbox_preds,
                                cfg,
                                scale_factor,
                                rescale=False):
        """
        |   YOLOv3   | lv5    | lv4    | lv3    | Total  |
        | ---------- | ------ | ------ | ------ | ------ |
        | pre-nms    | 0.28ms | 0.32ms | 0.29ms | 1.1ms  |
        | nms        | NA     | NA     | NA     | 13.9ms |

        Args:
            cls_scores: list(tensor), tensor <=> image.
            bbox_preds: list(tensor), tensor <=> image.
            cfg: test cfg.
            scale_factor:
            rescale:

        Returns:

        """
        # get the bbox for a single image.
        assert len(cls_scores) == len(bbox_preds)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
            # for a single level.
            bboxes, scores = bbox_pred, cls_score
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < cls_score.size(0):
                max_scores, _ = cls_score.max(dim=1)  # (h*w*num_anchor for all levels,)
                _, topk_inds = max_scores.topk(nms_pre)

                # in yolov3, bboxes is not delta but decoded box.
                bboxes, scores = bbox_pred[topk_inds, :], cls_score[topk_inds, :]
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes, dim=0)
        mlvl_scores = torch.cat(mlvl_scores, dim=0)

        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        # add background.
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)

        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels

    def loss(self,
             all_bboxes,
             all_box_centers,
             all_box_scales,
             all_class_pred,
             all_objectness,
             all_anchors,
             all_offsets,
             all_featmaps,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        box_preds = torch.cat(all_bboxes, dim=1)  # (batch, h*w*num_anchor for all levels, 4)
        all_preds = [torch.cat(p, dim=1) for p in [
            all_objectness, all_box_centers, all_box_scales, all_class_pred]]
        all_targets = self._target_generator(
            box_preds, gt_bboxes, gt_labels, all_anchors, all_offsets, all_featmaps, img_metas)
        obj_loss, center_loss, scale_loss, cls_loss = self._loss(*(all_preds + all_targets))
        return {'losses/yolo_loss_obj': obj_loss,
                'losses/yolo_loss_center': center_loss,
                'losses/yolo_loss_scale': scale_loss,
                'losses/yolo_loss_cls': cls_loss}


class YOLOV3TargetGenerator(nn.Module):
    """YOLOV3 target generator that merges the prefetched targets and dynamic targets."""

    def __init__(self, num_class, ignore_iou_thresh):
        super(YOLOV3TargetGenerator, self).__init__()
        self._num_class = num_class
        self._dynamic_target = YOLOV3DynamicTargetGeneratorSimple(num_class, ignore_iou_thresh)
        self._label_smooth = False

    def forward_single_image(self,
                             gt_boxes,
                             gt_labels,
                             img_metas,
                             shift_anchor_boxes,
                             shape_like,
                             num_anchors,
                             anchors,
                             pad_shape,
                             all_featmaps,
                             num_offsets):
        # shape_like: (h3*w3+h2*w2+h1*w1, 9 anchors, 2).
        center_targets = torch.zeros(shape_like).cuda()
        scale_targets = torch.zeros_like(center_targets)
        weights = torch.zeros_like(center_targets)
        objectness = torch.zeros_like(weights.split(1, dim=-1)[0])
        class_targets = torch.ones_like(objectness).repeat(1, 1, self._num_class) * -1

        gtx, gty, gtw, gth = point_to_center(gt_boxes, split=True, keep_axis=True)
        shift_gt_boxes = torch.cat((-0.5 * gtw, -0.5 * gth, 0.5 * gtw, 0.5 * gth), dim=-1)
        # ious between zero-center anchors(9) and zero-center gt boxes(gt num).
        ious = bbox_overlaps(shift_anchor_boxes, shift_gt_boxes)
        # assume the center of gt and anchor is aligned and find the best matched anchor scale.
        matches = ious.argmax(dim=0).to(torch.int32)  # (num_gt,)
        valid_gts = (gt_boxes >= 0).prod(dim=-1)  # (num_gt,)
        pad_height, pad_width = pad_shape
        for m in range(matches.shape[0]):
            # for each gt in a single image.
            if valid_gts[m] < 1:
                break
            match = matches[m]  # matched anchor idx, note that 0 <= match < 9.
            nlayer = np.nonzero(num_anchors > match)[0][0]
            height = all_featmaps[nlayer].shape[2]
            width = all_featmaps[nlayer].shape[3]
            mgtx, mgty, mgtw, mgth = (gtx[m, 0], gty[m, 0], gtw[m, 0], gth[m, 0])
            # compute the location of the gt top-left centers on the feature map level.
            loc_x = (mgtx / pad_width * width).to(torch.int32)
            loc_y = (mgty / pad_height * height).to(torch.int32)
            # write back to targets
            index = num_offsets[nlayer] + loc_y * width + loc_x
            center_targets[index, match, 0] = mgtx / pad_width * width - loc_x  # tx
            center_targets[index, match, 1] = mgty / pad_height * height - loc_y  # ty
            scale_targets[index, match, 0] = torch.log(max(mgtw, 1) / anchors[match, 0])
            scale_targets[index, match, 1] = torch.log(max(mgth, 1) / anchors[match, 1])
            weights[index, match, :] = 2.0 - mgtw * mgth / pad_width / pad_height
            first_n = img_metas.get('mixup_params', dict()).get('first_n_labels', len(matches))
            lambd = img_metas.get('mixup_params', dict()).get('lambd', 1.)
            if m < first_n:
                objectness[index, match, 0] = lambd
            else:
                objectness[index, match, 0] = 1. - lambd
            class_targets[index, match, :] = 0
            class_targets[index, match, int(gt_labels[m]) - 1] = 1
        return objectness, center_targets, scale_targets, weights, class_targets

    def slice(self, x, pad_num_anchors, pad_num_offsets):
        ret = []
        x = torch.stack(x, dim=0)  # stack a list of tensor from different images.
        for i in range(len(pad_num_anchors) - 1):
            y = x[:, pad_num_offsets[i]:pad_num_offsets[i + 1],
                pad_num_anchors[i]:pad_num_anchors[i + 1], :]
            ret.append(y.contiguous().view(y.size(0), y.size(1) * y.size(2), -1))
        return torch.cat(ret, dim=1)

    def forward(self,
                box_preds,
                gt_boxes,
                gt_labels,
                all_anchors,
                all_offsets,
                all_featmaps,
                img_metas):
        """

        Args:
            box_preds: (batch, h*w*num_anchor for all levels, 4).
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            all_anchors: list.
            all_offsets: list.
            all_featmaps: list.
            img_metas: list(dict).

        Returns:

        """
        assert len(all_offsets) == len(all_anchors) == len(all_featmaps)
        # anchors: (9, 2), offsets: (h3*w3+h2*w2+h1*w1, 2).
        anchors = torch.cat([a.view(-1, 2) for a in all_anchors], dim=0)
        offsets = torch.cat([o.view(-1, 2) for o in all_offsets], dim=0)
        # num_anchors: [3, 6, 9], num_offsets: [h3*w3, h2*w2, h1*w1]
        num_anchors = torch.cumsum(torch.tensor([a.numel() // 2 for a in all_anchors]).cuda(),
                                   dim=0, dtype=torch.int32)
        num_offsets = torch.cumsum(torch.tensor([o.numel() // 2 for o in all_offsets]).cuda(),
                                   dim=0, dtype=torch.int32)
        pad_num_anchors = torch.cat((torch.zeros(1, dtype=torch.int32).cuda(), num_anchors), dim=0)
        pad_num_offsets = torch.cat((torch.zeros(1, dtype=torch.int32).cuda(), num_offsets), dim=0)

        # orig image size. the images in a batch share the same pad shape.
        pad_shape = img_metas[0]['pad_shape'][0:2]

        # for each ground-truth, find the best matching anchor within the particular grid
        # for instance, center of object 1 reside in grid (3, 4) in (16, 16) feature map
        # then only the anchor in (3, 4) is going to be matched
        anchor_boxes = torch.cat((0 * anchors, anchors), dim=-1)  # zero center anchors
        shift_anchor_boxes = center_to_point(anchor_boxes, split=False, use_int=False)
        shape_like = (anchors.view(1, -1, 2) * offsets.view(-1, 1, 2)).size()

        objectness, center_targets, scale_targets, weights, class_targets = multi_apply(
            self.forward_single_image,
            gt_boxes,
            gt_labels,
            img_metas,
            shift_anchor_boxes=shift_anchor_boxes,
            shape_like=shape_like,
            num_anchors=num_anchors,
            anchors=anchors,
            pad_shape=pad_shape,
            all_featmaps=all_featmaps,
            num_offsets=pad_num_offsets
        )
        # since some stages won't see partial anchors, so we have to slice the correct targets
        pslice = partial(self.slice, pad_num_anchors=pad_num_anchors,
                         pad_num_offsets=pad_num_offsets)
        obj_t = pslice(objectness)
        centers_t = pslice(center_targets)
        scales_t = pslice(scale_targets)
        weights_t = pslice(weights)
        clas_t = pslice(class_targets)

        with torch.no_grad():
            # since we don not care about the predictions which has the objness between
            # ignore_iou_thresh~pos_iou_thresh, here we use mask to ignore them.
            dynamic_t = self._dynamic_target(box_preds, gt_boxes)
            # use fixed target to override dynamic targets.
            obj, centers, scales, weights, clas = zip(
                dynamic_t, [obj_t, centers_t, scales_t, weights_t, clas_t])
            mask = obj[1] > 0
            objectness = torch.where(mask, obj[1], obj[0])
            mask2 = mask.repeat(1, 1, 2)
            center_targets = torch.where(mask2, centers[1], centers[0])
            scale_targets = torch.where(mask2, scales[1], scales[0])
            weights = torch.where(mask2, weights[1], weights[0])
            mask3 = mask.repeat(1, 1, self._num_class)
            class_targets = torch.where(mask3, clas[1], clas[0])

            # TODO label smooth is disabled in gluoncv, have a try.
            if self._label_smooth:
                smooth_weight = 1. / self._num_class
                class_targets = torch.where(
                    class_targets > 0.5, class_targets - smooth_weight, class_targets)
                class_targets = torch.where(
                    class_targets < -0.5, class_targets,
                    torch.ones_like(class_targets) * smooth_weight)
            class_mask = mask3 * (class_targets >= 0)
            return [x.detach() for x in [objectness, center_targets, scale_targets,
                                         weights, class_targets, class_mask]]


class YOLOV3DynamicTargetGeneratorSimple(nn.Module):
    """YOLOV3 target generator that requires network predictions.
    `Dynamic` indicate that the targets generated depend on current network.
    `Simple` indicate that it only support `pos_iou_thresh` >= 1.0,
    otherwise it's a lot more complicated and slower.
    (box regression targets and class targets are not necessary when `pos_iou_thresh` >= 1.0)

    Parameters
    ----------
    num_class : int
        Number of foreground classes.
    ignore_iou_thresh : float
        Anchors that has IOU in `range(ignore_iou_thresh, pos_iou_thresh)` don't get
        penalized of objectness score.

    """

    def __init__(self, num_class, ignore_iou_thresh):
        super(YOLOV3DynamicTargetGeneratorSimple, self).__init__()
        self._num_class = num_class
        self._ignore_iou_thresh = ignore_iou_thresh

    def forward(self, box_preds, gt_boxes):
        """
        Parameters
        ----------
        box_preds: Predicted bounding boxes. (batch, xx, 4).
        gt_boxes: Ground-truth bounding boxes.

        Returns
        -------
        (tuple of) tensor.
            objectness: 0 for negative, 1 for positive, -1 for ignore. (batch, xx, 1).
            center_targets: regression target for center x and y. (batch, xx, 2).
            scale_targets: regression target for scale x and y. (batch, xx, 2).
            weights: element-wise gradient weights for center_targets and scale_targets.
            class_targets: a one-hot vector for classification. (batch, xx, 80).
        """
        with torch.no_grad():
            objness_t = torch.zeros_like(torch.unsqueeze(box_preds[:, :, 0], -1))
            center_t = torch.zeros_like(box_preds[:, :, 0:2])
            scale_t = torch.zeros_like(box_preds[:, :, 0:2])
            weight_t = torch.zeros_like(box_preds[:, :, 0:2])
            class_t = torch.ones_like(objness_t.repeat(1, 1, self._num_class)) * -1
            ious_max = []
            for box_preds_per_img, gt_boxes_per_img in zip(box_preds, gt_boxes):
                ious = bbox_overlaps(box_preds_per_img, gt_boxes_per_img)
                ious_max.append(torch.max(ious, dim=-1, keepdim=True)[0])  # (h*w*num_anchors, 1)
            ious_max = torch.stack(ious_max, dim=0)
            # use -1 for ignored.
            objness_t = (ious_max > self._ignore_iou_thresh).to(torch.float32) * -1
            return objness_t, center_t, scale_t, weight_t, class_t


class YOLOV3Loss(nn.Module):

    def __init__(self):  # TODO use __call__
        super(YOLOV3Loss, self).__init__()

    def forward(self, objness, box_centers, box_scales, cls_preds,
                objness_t, center_t, scale_t, weight_t, class_t, class_mask):
        # compute some normalization count, except batch-size
        denorm = torch.tensor(objness_t.size())[1:].prod().to(torch.float32)
        class_mask = class_mask.to(torch.float32)
        weight_t = weight_t * objness_t
        hard_objness_t = torch.where(
            objness_t > 0, torch.ones_like(objness_t), objness_t)
        new_objness_mask = torch.where(
            objness_t > 0, objness_t, (objness_t >= 0).to(torch.float32))
        obj_loss = F.binary_cross_entropy_with_logits(
            objness, hard_objness_t, new_objness_mask) * denorm
        center_loss = F.binary_cross_entropy_with_logits(
            box_centers, center_t, weight_t) * denorm * 2
        scale_loss = weighted_l1(box_scales, scale_t, weight_t, reduction='mean') * denorm * 2
        denorm_class = torch.tensor(class_t.size())[1:].prod(dtype=torch.float32)
        class_mask = class_mask * objness_t
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_preds, class_t, class_mask) * denorm_class
        return obj_loss, center_loss, scale_loss, cls_loss
