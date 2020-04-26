import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import multi_apply, nms_agnostic
from mmdet.core.utils.summary import add_summary
from mmdet.core.bbox import bbox_areas
from mmdet.models.losses import py_sigmoid_focal_loss
from mmdet.models.utils import Scale
from .anchor_head import AnchorHead
from ..registry import HEADS


@HEADS.register_module
class FCOSHead(AnchorHead):

    def __init__(self,
                 in_channel=256,
                 reg_in_channel=None,
                 fpn_strides=(8, 16, 32, 64, 128),
                 obj_sizes_of_interest=((-1, 64), (64, 128), (128, 256), (256, 512), (512, 99999)),
                 num_classes=81,
                 num_conv=4,
                 prior_prob=0.01,
                 norm_eval=True):
        super(AnchorHead, self).__init__()
        self.num_classes = num_classes - 1
        self.norm_eval = norm_eval
        self.fpn_strides = fpn_strides

        reg_in_channel = reg_in_channel if reg_in_channel else in_channel
        cls_tower, bbox_tower = [], []
        for i in range(num_conv):
            # individual 4 conv for cls and reg shared across all the levels.
            cls_tower.extend([
                nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(32, in_channel),
                nn.ReLU()
            ])
            bbox_tower.extend([
                nn.Conv2d(reg_in_channel, reg_in_channel, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(reg_in_channel // 8, reg_in_channel),
                nn.ReLU()
            ])

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(in_channel, self.num_classes,
                                    kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(reg_in_channel, 4,
                                   kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channel, 1,
                                    kernel_size=3, stride=1, padding=1)

        # initialize the bias for focal loss
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        # TODO different from origin paper.
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])
        # self.scales = nn.ModuleList([Scale(init_value=v) for v in [1., 2., 4., 8., 16.]])

        self._target_generator = FCOSTargetGenerator(obj_sizes_of_interest)
        self._loss = FCOSLoss(self.num_classes)

    def init_weights(self):
        for modules in [self.cls_tower, self.bbox_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    normal_init(l, std=0.01)
        normal_init(self.cls_logits, std=0.01, bias=self.bias_value)
        normal_init(self.bbox_pred, std=0.01)
        normal_init(self.centerness, std=0.01)

    def forward(self, feats):
        """
        FCOS-R50: R50 takes 22.5ms, FPN takes 2.95ms, HEAD takes 20.9ms, consuming 46.36ms.
                  location takes 0.9ms for each level, consuming 4.60ms.

        |     | cls_tower | bbox_tower | centerness | cls_logits | bbox_pred | location | Total   |
        | --- | --------- | ---------- | ---------- | ---------- | --------- | -------- | ------- |
        | P3  | 4.90ms    | 4.91ms     | 0.14ms     | 0.34ms     | 0.15ms    | 0.9ms    | 11.41ms |
        | P4  | 1.73ms    | 1.72ms     | NA         | NA         | NA        | 0.9ms    | 4.70ms  |
        | P5  | 1.00ms    | 1.09ms     | NA         | NA         | NA        | 0.9ms    | 3.33ms  |
        | P6  | NA        | NA         | NA         | NA         | NA        | 0.9ms    | 2.91ms  |
        | P7  | NA        | NA         | NA         | NA         | NA        | 0.9ms    | 2.97ms  |

        The GN takes a lot of time. In P3, conv (0.80ms) + gn (0.45ms) = 1.22ms.
        In RetinaNet-R50, HEAD takes 17.48ms.
        Args:
            feats: list(tensor).

        Returns:

        """
        if isinstance(feats, tuple) and isinstance(feats[0], tuple):
            feats, reg_feats = feats
        else:
            reg_feats = feats

        all_logits, all_bbox_reg, all_centerness = [], [], []
        for l, (feature, reg_feature) in enumerate(zip(feats, reg_feats)):
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(reg_feature)

            all_centerness.append(self.centerness(cls_tower))
            all_logits.append(self.cls_logits(cls_tower))
            all_bbox_reg.append(torch.exp(self.scales[l](self.bbox_pred(bbox_tower))))

        all_locations, _ = multi_apply(
            self.compute_locations_single_level,
            feats,
            self.fpn_strides,
        )
        return all_locations, all_logits, all_bbox_reg, all_centerness

    def compute_locations_single_level(self, feature, stride):
        """The locations are same for all the images in a batch."""
        h, w = feature.size()[-2:]
        device = feature.device

        shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2  # (hi * wi, 2)
        return locations, None

    def get_bboxes(self,
                   all_locations,
                   all_logits,
                   all_bbox_reg,
                   all_centerness,
                   img_metas,
                   cfg,
                   rescale=False):
        """
        |      | get_bbox | nms    | Total  |
        | ---- | -------- | ------ | ------ |
        |      | 6.0ms    | 20.2ms | 26.2ms |

        Args:
            all_locations: list(tensor). tensor <=> level, (hi * wi, 2). locations are fixed and
                only depends on the size of feature map.
            all_logits: list(tensor), tensor <=> level. (batch, 80, h, w).
            all_bbox_reg: list(tensor), tensor <=> level. (batch, 4, h, w).
            all_centerness: list(tensor), tensor <=> level. (batch, 1, h, w).
            img_metas: list(dict).
            cfg: test cfg.
            rescale:

        Returns:

        """
        # this takes 1.2ms for each level, consuming 6ms in total.
        bboxes_results, scores_results, labels_results = multi_apply(
            self.get_bboxes_single_level,
            all_locations,
            all_logits,
            all_bbox_reg,
            all_centerness,
            cfg=cfg
        )  # list(list(tensor)).

        # note that the cls results are split into scores(n,) and labels(n,).
        level_num = len(bboxes_results)
        imgs_per_gpu = len(bboxes_results[0])
        bboxes = [[] for _ in range(imgs_per_gpu)]
        scores = [[] for _ in range(imgs_per_gpu)]
        labels = [[] for _ in range(imgs_per_gpu)]
        for lvl in range(level_num):
            for idx in range(imgs_per_gpu):
                bboxes[idx].append(bboxes_results[lvl][idx].detach())
                scores[idx].append(scores_results[lvl][idx].detach())
                labels[idx].append(labels_results[lvl][idx].detach())

        result_list = []
        for idx in range(imgs_per_gpu):
            scale_factor = img_metas[idx]['scale_factor']
            bboxes_per_img = torch.cat(bboxes[idx], dim=0)
            scores_per_img = torch.cat(scores[idx], dim=0)
            labels_per_img = torch.cat(labels[idx], dim=0)

            if rescale:
                bboxes_per_img /= bboxes_per_img.new_tensor(scale_factor)
            # this takes 0.185ms for each class, consuming 20.15ms.
            det_bboxes, det_labels = nms_agnostic(
                bboxes_per_img, scores_per_img, labels_per_img,
                cfg.score_thr, cfg.nms, self.num_classes + 1, cfg.max_per_img)
            result_list.append((det_bboxes, det_labels))
        return result_list

    def get_bboxes_single_level(self,
                                locations,
                                box_cls,
                                box_regression,
                                centerness,
                                cfg):
        """

        Args:
            locations: tensor, (hi * wi, 2).
            box_cls: tensor, (batch, 80, h, w).
            box_regression: tensor, (batch, 4, h, w).
            centerness: tensor, (batch, 1, h, w).
            cfg: test cfg.

        Returns:
            bboxes_results: list(tensor), tensor <=> image, (hi * wi, 4).
            scores_results: list(tensor), tensor <=> image, (hi * wi,). cls scores of targets.
            labels_results: list(tensor), tensor <=> image, (hi * wi,). cls labels of targets.
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations. note that the view below will allocate
        # new tensor to prevent from modifying the origin tensor.
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1).reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1).reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1).reshape(N, -1).sigmoid()

        candidate_inds = box_cls > cfg.get('score_thr', 0)  # (batch, h * w, class_num)
        # (batch,). a location may be counted for multiple times.
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=cfg.get('nms_pre', 99999))

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        bboxes_results, scores_results, labels_results = [], [], []
        for i in range(N):
            # for a single image.
            per_box_cls = box_cls[i]  # (h * w, class_num)
            per_candidate_inds = candidate_inds[i]  # (h * w, class_num), value 0 or 1.
            per_box_cls = per_box_cls[per_candidate_inds]

            # note that a single position may correspond to more that one cls,
            # i.e. there may be more than one value > 0 in dim1.
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]  # (nonzero_num,), each in [0, h*w)
            per_class = per_candidate_nonzeros[:, 1] + 1  # (nonzero_num,), each in [1, class_num]

            per_box_regression = box_regression[i]  # (h * w, 4)
            per_box_regression = per_box_regression[per_box_loc]  # gather the chosen reg.
            per_locations = locations[per_box_loc]  # gather the chosen locations.

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)  # padded scale.

            bboxes_results.append(detections)
            scores_results.append(per_box_cls)
            labels_results.append(per_class)
        return bboxes_results, scores_results, labels_results

    def loss(self,
             all_locations,
             all_logits,
             all_bbox_reg,
             all_centerness,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        all_targets = self._target_generator(all_locations, gt_bboxes, gt_labels, img_metas)
        cls_loss, reg_loss, centerness_loss = self._loss(
            all_logits, all_bbox_reg, all_centerness, *all_targets,
            gamma=cfg.gamma, alpha=cfg.alpha)
        return {'losses/fcos_loss_cls': cls_loss,
                'losses/fcos_loss_reg': reg_loss,
                'losses/fcos_loss_center': centerness_loss}

    def train(self, mode=True):
        super(FCOSHead, self).train(mode)
        if mode and self.norm_eval:
            for _, m in self.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


class FCOSTargetGenerator(nn.Module):

    def __init__(self, obj_sizes_of_interest):
        super(FCOSTargetGenerator, self).__init__()
        self.obj_sizes_of_interest = obj_sizes_of_interest
        self.INF = 1e8
        self.summary = None

    def reset_summary(self):
        self.summary = {'in_boxes_all': 0, 'in_boxes_care': 0, 'num_pos': 0, 'num_neg': 0}

    def forward_single_image(self,
                             gt_boxes,
                             gt_labels,
                             locations,
                             obj_sizes_of_interest):
        """Return the cls and reg target for an image.

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt, 4).
            locations: tensor, same for all images in a batch, (h*w for all levels, 2).
            obj_sizes_of_interest: same for all images in a batch, (h*w for all levels, 2).

        Returns:
            labels: tensor, tensor <=> img, (h*w for all levels,).
            reg_targets: tensor, tensor <=> img, (h*w for all levels, 4).
        """
        # get reg target first since we need the information to judge whether a point is
        # in sizes of interest or not.
        locations_x, locations_y = locations[:, 0], locations[:, 1]
        # broadcast, (h*w for all levels, 1) - (1, gt_num)
        l = locations_x[:, None] - gt_boxes[:, 0][None]
        t = locations_y[:, None] - gt_boxes[:, 1][None]
        r = gt_boxes[:, 2][None] - locations_x[:, None]
        b = gt_boxes[:, 3][None] - locations_y[:, None]
        reg_targets = torch.stack([l, t, r, b], dim=2)  # (h*w for all levels, gt_num, 4)

        is_in_boxes = reg_targets.min(dim=2)[0] > 0  # (h*w for all levels, num_gt)
        max_reg_targets = reg_targets.max(dim=2)[0]
        # limit the regression range for each location
        is_cared_in_the_level = \
            (max_reg_targets >= obj_sizes_of_interest[:, [0]]) & \
            (max_reg_targets <= obj_sizes_of_interest[:, [1]])  # (h*w for all levels, num_gt)

        area = bbox_areas(gt_boxes)
        locations_to_gt_area = area[None].repeat(len(locations), 1)  # (h*w for all levels, num_gt)
        locations_to_gt_area[is_in_boxes == 0] = self.INF
        locations_to_gt_area[is_cared_in_the_level == 0] = self.INF

        # if there are still more than one object for a location, we choose the one with
        # minimal area.
        locations_to_min_aera, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

        # (h*w for all levels, gt_num, 4) => (h*w for all levels, 4)
        reg_targets = reg_targets[range(len(locations)), locations_to_gt_inds]
        labels = gt_labels[locations_to_gt_inds]  # (h*w for all levels,)
        labels[locations_to_min_aera == self.INF] = 0

        self.summary['in_boxes_all'] += is_in_boxes.sum().item()
        self.summary['in_boxes_care'] += (is_in_boxes & is_cared_in_the_level).sum().item()
        self.summary['num_pos'] += (labels > 0).sum().item()
        self.summary['num_neg'] += (labels == 0).sum().item()

        return labels, reg_targets

    def forward(self,
                all_locations,
                gt_boxes,
                gt_labels,
                img_metas):
        """

        Args:
            all_locations: list(tensor). tensor <=> level, (hi * wi, 2). locations are fixed and
                only depends on the size of feature map.
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:
            labels_level_first: list(tensor). tensor <=> level, (batch * hi * wi,).
            reg_targets_level_first: list(tensor). tensor <=> level, (batch * hi * wi, 4).
        """
        self.reset_summary()

        expanded_obj_sizes_of_interest = []
        for i, locations_per_level in enumerate(all_locations):
            obj_sizes_of_interest_per_level = locations_per_level.new_tensor(
                self.obj_sizes_of_interest[i])
            expanded_obj_sizes_of_interest.append(
                # (2,) => (hi * wi, 2)
                obj_sizes_of_interest_per_level[None].expand(len(locations_per_level), -1)
            )

        num_points_per_level = [len(locations_per_level) for locations_per_level in all_locations]
        locations_all_level = torch.cat(all_locations, dim=0)
        # (h*w for all levels, 2), indicating the size range of a location.
        expanded_obj_sizes_of_interest = torch.cat(expanded_obj_sizes_of_interest, dim=0)

        labels, reg_targets = multi_apply(
            self.forward_single_image,
            gt_boxes,
            gt_labels,
            locations=locations_all_level,
            obj_sizes_of_interest=expanded_obj_sizes_of_interest,
        )

        add_summary('fcos_head', average_factor=len(gt_boxes), **self.summary)

        with torch.no_grad():
            for i in range(len(labels)):
                # for each image.
                labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
                reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

            labels_level_first, reg_targets_level_first = [], []
            for level in range(len(all_locations)):
                labels_level_first.append(
                    torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0).detach())
                reg_targets_level_first.append(
                    torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets],
                    dim=0).detach())

            return labels_level_first, reg_targets_level_first


class FCOSLoss(object):

    def __init__(self, num_class):
        self.num_class = num_class

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def cls_loss_func(self, *args, **kwargs):
        return py_sigmoid_focal_loss(*args, **kwargs)

    def box_reg_loss_func(self, pred, target, weight=None):
        pred_left, pred_top, pred_right, pred_bottom = pred[:, 0], pred[:, 1], \
                                                       pred[:, 2], pred[:, 3]
        target_left, target_top, target_right, target_bottom = target[:, 0], target[:, 1], \
                                                               target[:, 2], target[:, 3]
        target_aera = (target_left + target_right) * (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()

    def centerness_loss_func(self, *args, **kwargs):
        return F.binary_cross_entropy_with_logits(*args, **kwargs)

    def __call__(self,
                 all_logits,
                 all_bbox_reg,
                 all_centerness,
                 label_targets,
                 reg_targets,
                 gamma,
                 alpha):
        """

        Args:
            all_logits: list(tensor), tensor <=> level. (batch, 80, h, w).
            all_bbox_reg: list(tensor), tensor <=> level. (batch, 4, h, w).
            all_centerness: list(tensor), tensor <=> level. (batch, 1, h, w).
            label_targets: list(tensor). tensor <=> level, (batch * h * w,).
            reg_targets: list(tensor). tensor <=> level, (batch * h * w, 4).

        Returns:

        """
        N = all_logits[0].size(0)
        box_cls_flatten, box_regression_flatten, centerness_flatten, \
                labels_flatten, reg_targets_flatten = [], [], [], [], []

        for i in range(len(label_targets)):
            # for a single level.
            # (batch, xx, h, w) to (batch * h * w, xx), same as label_targets and reg_targets.
            box_cls_flatten.append(all_logits[i].permute(0, 2, 3, 1).reshape(-1, self.num_class))
            box_regression_flatten.append(all_bbox_reg[i].permute(0, 2, 3, 1).reshape(-1, 4))
            centerness_flatten.append(all_centerness[i].reshape(-1))
            labels_flatten.append(label_targets[i])
            reg_targets_flatten.append(reg_targets[i])

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)  # (batch*h*w for all levels, 80)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)  # each in [0, num_class], including bg.
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        # note that the shape of logits is (xx, 80) and the origin shape of one hot tensor
        # is (xx, 81). if there is a bg bbox, we except the values in logit[bg_idx, :]
        # are closed to 0.
        shape_like = labels_flatten[:, None].repeat(1, self.num_class + 1)
        labels_flatten_one_hot = torch.zeros_like(
            shape_like).scatter_(1, labels_flatten[:, None], torch.ones_like(shape_like))[:, 1:]

        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten_one_hot,
            torch.ones_like(labels_flatten_one_hot, dtype=torch.float32),
            gamma=gamma,
            alpha=alpha,
            reduction='sum',
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss
