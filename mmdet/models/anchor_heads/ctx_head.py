import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import normal_init
from mmdet.core import multi_apply, multiclass_nms
from mmdet.core.bbox import bbox_overlaps
from mmdet.core.utils.common import gather_feat, tranpose_and_gather_feat
from mmdet.models.losses import ct_focal_loss, weighted_l1, smooth_l1_loss, giou_loss
from mmdet.models.utils import (
    build_norm_layer, gaussian_radius, draw_umich_gaussian, ConvModule, simple_nms,
    build_conv_layer, ShortcutConv2d, draw_truncate_gaussian)
from mmdet.ops import ModulatedDeformConvPack, ModulatedDeformConv
from .anchor_head import AnchorHead
from ..registry import HEADS


@HEADS.register_module
class CTXHead(AnchorHead):

    def __init__(self,
                 planes=(128, 64),
                 hm_head_channels=((128, 128), (64, 64)),
                 wh_head_channels=((32, 32), (32, 32)),
                 reg_head_channels=((32, 32), (32, 32)),
                 num_classes=81,
                 use_dla=False,
                 conv_cfg=None,
                 hm_init_value=-2.19,
                 length_range=((64, 512), (1, 64)),
                 down_ratio=(8, 4),
                 fast_nms=True,
                 trand_nms=False,
                 hm_weight=(1., 1.),
                 wh_weight=(0.1, 0.1),
                 off_weight=(1., 1.)):
        super(AnchorHead, self).__init__()
        self.planes = planes
        self.num_classes = num_classes
        self.num_fg = num_classes - 1

        self.use_dla = use_dla
        self.conv_cfg = conv_cfg

        self.hm_init_value = hm_init_value
        self.down_ratio = down_ratio
        self.fast_nms = fast_nms
        self.trand_nms = trand_nms
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.off_weight = off_weight

        # heads
        hm, wh, reg = multi_apply(
            self._init_branch_layers,
            self.planes,
            hm_head_channels,
            wh_head_channels,
            reg_head_channels
        )

        self.hm = nn.ModuleList(hm)
        self.wh = nn.ModuleList(wh)
        self.reg = nn.ModuleList(reg)

        self._target_generator = CTTargetGenerator(self.down_ratio, length_range, self.num_fg)
        self._loss = CTLoss(self.hm_weight, self.wh_weight, self.off_weight)

    def _init_branch_layers(self, planes, hm_chan, wh_chan, reg_chan):
        hm_layers, wh_layers, reg_layers = [], [], []
        inp = planes
        for outp in hm_chan:
            hm_layers.append(
                ConvModule(
                    inp,
                    outp,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg))
            inp = outp
        hm_layers.append(nn.Conv2d(inp, self.num_fg, 1))

        inp = planes
        for outp in wh_chan:
            wh_layers.append(
                ConvModule(
                    inp,
                    outp,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg))
            inp = outp
        wh_layers.append(nn.Conv2d(inp, 2, 1))

        inp = planes
        for outp in reg_chan:
            reg_layers.append(
                ConvModule(
                    inp,
                    outp,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg))
            inp = outp
        reg_layers.append(nn.Conv2d(inp, 2, 1))

        hm_layers = nn.Sequential(*hm_layers)
        wh_layers = nn.Sequential(*wh_layers)
        reg_layers = nn.Sequential(*reg_layers)
        return hm_layers, wh_layers, reg_layers

    def init_weights(self):
        for hm in self.hm:
            for m in hm.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)

            hm[-1].bias.data.fill_(self.hm_init_value)

        for wh in self.wh:
            for m in wh.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

        for reg in self.reg:
            for m in reg.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward_single(self, x, hm, wh, reg):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 2, h, w).
            reg: Tensor, (batch, 2, h, w).
        """
        f_hm = hm(x)
        f_wh = wh(x)
        f_reg = reg(x)
        return f_hm, f_wh, f_reg

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 2, h, w).
            reg: None or tensor, (batch, 2, h, w).
        """
        hms, whs, regs = multi_apply(
            self.forward_single,
            feats[::-1],
            self.hm,
            self.wh,
            self.reg
        )

        return hms, whs, regs

    def get_bboxes_single(self,
                          pred_heatmap,
                          pred_wh,
                          pred_reg_offset,
                          down_ratio,
                          topk=100):
        batch, cat, height, width = pred_heatmap.size()
        pred_heatmap = pred_heatmap.detach()
        wh = pred_wh.detach()
        reg = pred_reg_offset.detach()

        # perform nms on heatmaps
        heat = simple_nms(pred_heatmap.sigmoid_())  # used maxpool to filter the max score

        # (batch, topk)
        scores, inds, clses, ys, xs = self._topk(heat, topk=topk)
        if reg is not None:
            reg = tranpose_and_gather_feat(reg, inds)  # (batch, topk, 2)
            reg = reg.view(batch, topk, 2)
            xs = xs.view(batch, topk, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, topk, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, topk, 1) + 0.5
            ys = ys.view(batch, topk, 1) + 0.5

        wh = tranpose_and_gather_feat(wh, inds)  # (batch, topk, 2)
        wh = wh.view(batch, topk, 2)
        clses = clses.view(batch, topk, 1).float()
        scores = scores.view(batch, topk, 1)
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2) * down_ratio
        return clses, scores, bboxes

    def get_bboxes(self,
                   pred_heatmap,
                   pred_wh,
                   pred_reg_offset,
                   img_metas,
                   cfg,
                   rescale=False):
        clses, scores, bboxes = multi_apply(
            self.get_bboxes_single,
            pred_heatmap,
            pred_wh,
            pred_reg_offset,
            self.down_ratio,
            topk=getattr(cfg, 'max_per_img', 100)
        )

        result_list = []
        score_thr = getattr(cfg, 'score_thr', 0.01)
        for batch_i in range(bboxes[0].shape[0]):
            scores_per_img = [score[batch_i] for score in scores]
            scores_keep = [(score_per > score_thr).squeeze(-1) for score_per in scores_per_img]

            scores_per_img = [score_per[score_keep] for (score_per, score_keep)
                              in zip(scores_per_img, scores_keep)]
            bboxes_per_img = [bbox[batch_i][score_keep] for (bbox, score_keep)
                              in zip(bboxes, scores_keep)]
            labels_per_img = [cls[batch_i][score_keep].squeeze(-1).long() for (cls, score_keep)
                              in zip(clses, scores_keep)]

            img_shape = img_metas[batch_i]['pad_shape']
            if self.fast_nms and len(scores_per_img) > 1:
                for i in range(len(bboxes_per_img)):
                    bboxes_per_img[i][:, 0::2] = bboxes_per_img[i][:, 0::2].clamp(
                        min=0, max=img_shape[1] - 1)
                    bboxes_per_img[i][:, 1::2] = bboxes_per_img[i][:, 1::2].clamp(
                        min=0, max=img_shape[0] - 1)

                b1_keeps, b2_keeps = [], []
                for idx in range(len(scores_per_img) - 1):
                    bboxes_b1_per_img, bboxes_b2_per_img = bboxes_per_img[idx], \
                                                           bboxes_per_img[idx + 1]
                    labels_b1_per_img, labels_b2_per_img = labels_per_img[idx], \
                                                           labels_per_img[idx + 1]
                    scores_b1_per_img, scores_b2_per_img = scores_per_img[idx], \
                                                           scores_per_img[idx + 1]
                    duplicate_cls_b1 = bboxes_b1_per_img.new_ones(
                        (bboxes_b1_per_img.shape[0], bboxes_b2_per_img.shape[0]),
                        dtype=torch.long) * labels_b1_per_img.unsqueeze(-1)
                    duplicate_cls_b2 = bboxes_b1_per_img.new_ones(
                        (bboxes_b1_per_img.shape[0], bboxes_b2_per_img.shape[0]),
                        dtype=torch.long) * labels_b2_per_img.unsqueeze(0)
                    duplicate_cls = (duplicate_cls_b1 == duplicate_cls_b2)

                    b1_keep = bboxes_b1_per_img.new_ones((bboxes_b1_per_img.shape[0],),
                                                         dtype=torch.bool)
                    b2_keep = bboxes_b2_per_img.new_ones((bboxes_b2_per_img.shape[0],),
                                                         dtype=torch.bool)
                    if duplicate_cls.any():
                        ious_large = bbox_overlaps(bboxes_b1_per_img,
                                                   bboxes_b2_per_img) > 0.6
                        duplicate = ious_large & duplicate_cls
                        scores_b2_max, b2_max_loc = (scores_b2_per_img.view(
                            1, -1) * duplicate.float()).max(1)
                        b1_keep = (scores_b1_per_img.view(-1) >= scores_b2_max)
                        b2_max_loc_keep = b2_max_loc[~b1_keep]
                        b2_max_loc_keep_hot = bboxes_b1_per_img.new_zeros((duplicate.shape[1],),
                                                                          dtype=torch.bool)
                        b2_max_loc_keep_hot.scatter(0, b2_max_loc_keep, 1)
                        b2_keep = ~(b1_keep.view(-1, 1) * duplicate).any(0)
                        b2_keep = b2_keep | b2_max_loc_keep_hot

                    b1_keeps.append(b1_keep)
                    b2_keeps.append(b2_keep)

                level_keep = [b1_keeps[0]]
                for idx in range(len(self.down_ratio) - 2):
                    level_keep.append(b1_keeps[idx + 1] & b2_keeps[idx])
                level_keep.append(b2_keeps[-1])

                scores_per_img = torch.cat([score_per_img[keep] for (score_per_img, keep)
                                            in zip(scores_per_img, level_keep)], dim=0)
                bboxes_per_img = torch.cat([bbox_per_img[keep] for (bbox_per_img, keep)
                                            in zip(bboxes_per_img, level_keep)], dim=0)
                labels_per_img = torch.cat([label_per_img[keep] for (label_per_img, keep)
                                            in zip(labels_per_img, level_keep)], dim=0)
            else:
                if len(scores_per_img) == 1:
                    scores_per_img = scores_per_img[0]
                    bboxes_per_img = bboxes_per_img[0]
                    labels_per_img = labels_per_img[0]
                else:
                    scores_per_img = torch.cat(scores_per_img, dim=0)
                    bboxes_per_img = torch.cat(bboxes_per_img, dim=0)
                    labels_per_img = torch.cat(labels_per_img, dim=0)

                bboxes_per_img[:, 0::2] = bboxes_per_img[:, 0::2].clamp(
                    min=0, max=img_shape[1] - 1)
                bboxes_per_img[:, 1::2] = bboxes_per_img[:, 1::2].clamp(
                    min=0, max=img_shape[0] - 1)

            if rescale:
                scale_factor = img_metas[batch_i]['scale_factor']
                bboxes_per_img /= bboxes_per_img.new_tensor(scale_factor)

            if self.trand_nms:
                label_hard = bboxes_per_img.new_zeros(bboxes_per_img.shape[0], self.num_classes)
                label_hard.scatter_(1, labels_per_img.view(-1, 1) + 1, 1)
                scores_per_img = scores_per_img * label_hard.float()
                bboxes_per_img, labels_per_img = multiclass_nms(
                    bboxes_per_img, scores_per_img, score_thr=score_thr,
                    nms_cfg=dict(type='nms', iou_thr=0.6), max_num=100)
            else:
                bboxes_per_img = torch.cat([bboxes_per_img, scores_per_img], dim=1)
                labels_per_img = labels_per_img.float()

            result_list.append((bboxes_per_img, labels_per_img))

        return result_list

    def loss(self,
             pred_heatmap,
             pred_wh,
             pred_reg_offset,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        all_targets = self._target_generator(gt_bboxes, gt_labels, img_metas)
        hm_loss, wh_loss, off_loss = self._loss(
            pred_heatmap, pred_wh, pred_reg_offset, *all_targets)

        loss_dict = dict()
        for idx, down_ratio in enumerate(self.down_ratio):
            loss_dict['losses/ctx_loss_hm_s{}'.format(down_ratio)] = hm_loss[idx]
            loss_dict['losses/ctx_loss_wh_s{}'.format(down_ratio)] = wh_loss[idx]
            loss_dict['losses/ctx_loss_reg_s{}'.format(down_ratio)] = off_loss[idx]

        return loss_dict

    def _topk(self, scores, topk):
        batch, cat, height, width = scores.size()

        # both are (batch, 80, topk)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), topk)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # both are (batch, topk). select topk from 80*topk
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
        topk_clses = (topk_ind / topk).int()
        topk_inds = gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, topk)
        topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, topk)
        topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, topk)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


class CTTargetGenerator(object):

    def __init__(self,
                 down_ratio,
                 length_range,
                 num_fg=80,
                 max_objs=128):
        self.num_fg = num_fg
        self.down_ratio = down_ratio
        self.length_range = length_range
        self.max_objs = max_objs

    def target_single_image(self, gt_boxes, gt_labels, pad_shape, boxes_area=None):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            pad_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            wh: tensor, tensor <=> img, (max_obj, 2).
            reg_mask: tensor, tensor <=> img, (max_obj,).
            ind: tensor, tensor <=> img, (max_obj,).
            reg: tensor, tensor <=> img, (max_obj, 2).
            center_location: tensor or None, tensor <=> img, (max_obj, 2).
        """
        boxes_areas = self.bbox_areas(gt_boxes)

        heatmap, wh, reg_mask, ind, reg = [], [], [], [], []
        output_hs, output_ws, gt_level_idx = [], [], []
        for i, (down_ratio) in enumerate(self.down_ratio):
            output_h, output_w = [shape // down_ratio for shape in pad_shape]
            heatmap.append(gt_boxes.new_zeros((self.num_fg, output_h, output_w)))

            wh.append(gt_boxes.new_zeros((self.max_objs, 2)))
            reg_mask.append(gt_boxes.new_zeros((self.max_objs,), dtype=torch.uint8))
            ind.append(gt_boxes.new_zeros((self.max_objs,), dtype=torch.long))
            reg.append(gt_boxes.new_zeros((self.max_objs, 2)))

            output_hs.append(output_h)
            output_ws.append(output_w)
            gt_level_idx.append(
                (boxes_areas >= self.length_range[i][0] ** 2) &
                (boxes_areas <= self.length_range[i][1] ** 2))

        heatmap, wh, reg_mask, ind, reg = multi_apply(
            self.target_single_single,
            heatmap,
            wh,
            reg_mask,
            ind,
            reg,
            gt_level_idx,
            output_hs,
            output_ws,
            self.down_ratio,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels)
        return heatmap, wh, reg_mask, ind, reg

    def target_single_single(self,
                             heatmap,
                             wh,
                             reg_mask,
                             ind,
                             reg,
                             gt_level_idx,
                             output_h,
                             output_w,
                             down_ratio,
                             gt_boxes,
                             gt_labels):
        gt_boxes = gt_boxes[gt_level_idx]
        gt_labels = gt_labels[gt_level_idx]

        gt_boxes /= down_ratio
        gt_boxes[:, [0, 2]] = torch.clamp(gt_boxes[:, [0, 2]], 0, output_w - 1)
        gt_boxes[:, [1, 3]] = torch.clamp(gt_boxes[:, [1, 3]], 0, output_h - 1)
        hs, ws = (gt_boxes[:, 3] - gt_boxes[:, 1], gt_boxes[:, 2] - gt_boxes[:, 0])

        for k in range(gt_boxes.shape[0]):
            cls_id = gt_labels[k] - 1
            h, w = hs[k], ws[k]
            if h > 0 and w > 0:
                center = gt_boxes.new_tensor([(gt_boxes[k, 0] + gt_boxes[k, 2]) / 2,
                                              (gt_boxes[k, 1] + gt_boxes[k, 3]) / 2])

                # no peak will fall between pixels
                ct_int = center.to(torch.int)
                radius = gaussian_radius((h.ceil(), w.ceil()))
                radius = max(0, int(radius.item()))
                draw_umich_gaussian(heatmap[cls_id], ct_int, radius)
                # directly predict the width and height
                wh[k] = wh.new_tensor([1. * w, 1. * h])
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = center - ct_int.float()
                reg_mask[k] = 1

        return heatmap, wh, reg_mask, ind, reg

    def bbox_areas(self, bboxes, keep_axis=False):
        x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], \
                                     bboxes[:, 2], bboxes[:, 3]
        areas = (y_max - y_min + 1) * (x_max - x_min + 1)
        if keep_axis:
            return areas[:, None]
        return areas

    def __call__(self, gt_boxes, gt_labels, img_metas):
        """

        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:

        """
        pad_shape = (img_metas[0]['pad_shape'][0], img_metas[0]['pad_shape'][1])
        heatmap, wh, reg_mask, ind, reg = multi_apply(
            self.target_single_image,
            gt_boxes,
            gt_labels,
            pad_shape=pad_shape
        )

        with torch.no_grad():
            s_heatmap, s_wh, s_reg_mask, s_ind, s_reg = [], [], [], [], []
            for level_i in range(len(heatmap[0])):
                s_heatmap.append(torch.stack([h[level_i] for h in heatmap], dim=0).detach())
                s_wh.append(torch.stack([t[level_i] for t in wh], dim=0).detach())
                s_reg_mask.append(torch.stack([t[level_i] for t in reg_mask], dim=0).detach())
                s_ind.append(torch.stack([t[level_i] for t in ind], dim=0).detach())
                s_reg.append(torch.stack([t[level_i] for t in reg], dim=0).detach())

            return s_heatmap, s_wh, s_reg_mask, s_ind, s_reg


class CTLoss(object):

    def __init__(self,
                 hm_weight=1.,
                 wh_weight=0.1,
                 off_weight=1.):
        super(CTLoss, self).__init__()
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.off_weight = off_weight

    def loss_single(self,
                    pred_hm,
                    pred_wh,
                    pred_reg_offset,
                    heatmap,
                    wh,
                    reg_mask,
                    ind,
                    reg_offset,
                    hm_weight,
                    wh_weight,
                    off_weight):
        H, W = pred_hm.shape[2:]
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        hm_loss = ct_focal_loss(pred_hm, heatmap) * hm_weight

        # (batch, 2, h, w) => (batch, max_obj, 2)
        pred = tranpose_and_gather_feat(pred_wh, ind)
        mask = reg_mask.unsqueeze(2).expand_as(pred).float()
        avg_factor = mask.sum() + 1e-4

        wh_loss = weighted_l1(pred, wh, mask, avg_factor=avg_factor) * wh_weight

        pred_reg = tranpose_and_gather_feat(pred_reg_offset, ind)
        off_loss = weighted_l1(pred_reg, reg_offset, mask,
                               avg_factor=avg_factor) * off_weight

        return hm_loss, wh_loss, off_loss

    def __call__(self,
                 pred_hm,
                 pred_wh,
                 pred_reg_offset,
                 heatmap,
                 wh,
                 reg_mask,
                 ind,
                 reg_offset):
        """

        Args:
            pred_hm: tensor, (batch, 80, h, w).
            pred_wh: tensor, (batch, 2, h, w).
            pred_reg_offset: None or tensor, (batch, 2, h, w).
            heatmap: tensor, (batch, 80, h, w).
            wh: tensor, (batch, max_obj, 2).
            reg_mask: tensor, tensor <=> img, (batch, max_obj).
            ind: tensor, (batch, max_obj).
            reg_offset: tensor, (batch, max_obj, 2).

        Returns:

        """
        hm_loss, wh_loss, off_loss = multi_apply(
            self.loss_single,
            pred_hm,
            pred_wh,
            pred_reg_offset,
            heatmap,
            wh,
            reg_mask,
            ind,
            reg_offset,
            self.hm_weight,
            self.wh_weight,
            self.off_weight
        )

        return hm_loss, wh_loss, off_loss
