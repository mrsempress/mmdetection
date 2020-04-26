import numpy as np
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.ops import ModulatedDeformConvPack, ModulatedDeformConv, DeformConvPack
from mmdet.core import multi_apply, nms_agnostic
from mmdet.core.utils.common import gather_feat, tranpose_and_gather_feat
from mmdet.core.utils.summary import (
    add_summary, every_n_local_step, add_feature_summary, add_histogram_summary)
from mmdet.models.losses import ct_focal_loss, weighted_l1, smooth_l1_loss, giou_loss
from mmdet.models.utils import (
    build_norm_layer, gaussian_radius, draw_umich_gaussian, extra_path, extra_mask_loss,
    ConvModule)
from .anchor_head import AnchorHead
from ..registry import HEADS


@HEADS.register_module
class MLCTHead(AnchorHead):

    def __init__(self,
                 inplane=256,
                 fpn_strides=(8, 16, 32),
                 head_conv=256,
                 stacked_convs=1,
                 obj_sizes_of_interest=((-1, 128), (96, 384), (256, 99999)),
                 num_classes=81,
                 use_smooth_l1=False,
                 use_neg_wh=False,
                 use_exp_wh=False,
                 use_giou=False,
                 use_nms=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 norm_eval=True,
                 select_feat_index=None,
                 hm_init_value=-2.19,
                 wh_weight=0.1,
                 hm_weight=1.,
                 gamma=2.):
        super(AnchorHead, self).__init__()
        self.inplane = inplane
        self.fpn_strides = fpn_strides
        self.head_conv = head_conv
        self.stacked_convs = stacked_convs
        self.use_neg_wh = use_neg_wh
        self.use_exp_wh = use_exp_wh
        self.use_nms = use_nms
        self.num_classes = num_classes - 1
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.select_feat_index = select_feat_index

        self.hm_init_value = hm_init_value
        self.wh_weight = wh_weight
        self.hm_weight = hm_weight

        # heads, share across all levels.
        self.wh = self._make_conv_layer(2)
        self.hm = self._make_conv_layer(self.num_classes)

        self._target_generator = MLCTTargetGenerator(
            self.num_classes, fpn_strides, obj_sizes_of_interest, use_giou)
        self._loss = MLCTLoss(use_smooth_l1, use_giou, wh_weight, hm_weight, gamma)

    def _make_conv_layer(self, num_output):
        conv_layers = []
        for i in range(self.stacked_convs):
            in_chn = self.inplane if i == 0 else self.head_conv
            conv_layers.append(
                ConvModule(
                    in_chn,
                    self.head_conv,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        conv_layers.append(nn.Conv2d(self.head_conv, num_output, 1))
        return nn.Sequential(*conv_layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        self.hm[-1].bias.data.fill_(self.hm_init_value)

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hms: list(tensor), tensor <=> level. (batch, 80, h, w).
            whs: list(tensor), tensor <=> level. (batch, 2, h, w).
        """
        hms, whs = [], []
        if self.select_feat_index:
            feats = [feats[i] for i in self.select_feat_index]

        for feat in feats:
            hm = self.hm(feat)
            wh = self.wh(feat)
            if self.use_neg_wh:
                wh = wh * -1
            if self.use_exp_wh:
                wh = wh.exp()
            hms.append(hm)
            whs.append(wh)

        if every_n_local_step(500):
            for lvl, (feat, hm, wh) in enumerate(zip(feats, hms, whs)):
                add_histogram_summary('mlct_head_feat_fpn_lv{}'.format(lvl), feat.detach().cpu())
                add_histogram_summary('mlct_head_feat_heatmap_lv{}'.format(lvl), hm.detach().cpu())
                add_histogram_summary('mlct_head_feat_wh_lv{}'.format(lvl), wh.detach().cpu())

            hm_summary = self.hm[-1]
            wh_summary = self.wh[-1]
            add_histogram_summary('mlct_head_param_hm', hm_summary.weight.detach().cpu(),
                                  is_param=True)
            add_histogram_summary('mlct_head_param_wh', wh_summary.weight.detach().cpu(),
                                  is_param=True)

            add_histogram_summary('mlct_head_param_hm_grad',
                                  hm_summary.weight.grad.detach().cpu(), collect_type='none')
            add_histogram_summary('mlct_head_param_wh_grad',
                                  wh_summary.weight.grad.detach().cpu(), collect_type='none')

        return hms, whs

    def get_bboxes(self,
                   pred_heatmap,
                   pred_wh,
                   img_metas,
                   cfg,
                   rescale=False):
        pred_heatmap = [x.detach().sigmoid_() for x in pred_heatmap]
        wh = [x.detach() for x in pred_wh]

        level_num = len(pred_heatmap)
        batch, cat, height, width = pred_heatmap[0].size()

        all_clses, all_scores, all_bboxes = [], [], []
        for lvl in range(level_num):
            heat_per_lvl = self._nms(pred_heatmap[lvl])

            topk = getattr(cfg, 'max_per_level', 100)
            # (batch, topk)
            scores, inds, clses, ys, xs = self._topk(heat_per_lvl, topk=topk)
            xs = xs.view(batch, topk, 1) + 0.5
            ys = ys.view(batch, topk, 1) + 0.5

            wh_per_lvl = tranpose_and_gather_feat(wh[lvl], inds)  # (batch, topk, 2)
            wh_per_lvl = wh_per_lvl.view(batch, topk, 2)
            clses = clses.view(batch, topk, 1).float()
            scores = scores.view(batch, topk, 1)
            bboxes = torch.cat([xs - wh_per_lvl[..., 0:1] / 2,
                                ys - wh_per_lvl[..., 1:2] / 2,
                                xs + wh_per_lvl[..., 0:1] / 2,
                                ys + wh_per_lvl[..., 1:2] / 2], dim=2)
            bboxes *= self.fpn_strides[lvl]
            all_clses.append(clses)
            all_scores.append(scores)
            all_bboxes.append(bboxes)

        # (batch, topk, xx) for all levels => (batch, topk*level_num, xx)
        all_clses, all_scores, all_bboxes = [torch.cat(x, dim=1)
                                             for x in [all_clses, all_scores, all_bboxes]]

        if not self.use_nms:
            topk_img = getattr(cfg, 'max_per_img', 100)
            all_scores, topk_inds = torch.topk(all_scores, topk_img, dim=1)
            all_clses = all_clses.gather(dim=1, index=topk_inds)
            all_bboxes = all_bboxes.gather(dim=1, index=topk_inds.repeat(1, 1, 4))

        result_list = []
        for idx in range(bboxes.shape[0]):
            bboxes_per_img = all_bboxes[idx]
            scores_per_img = all_scores[idx]
            labels_per_img = all_clses[idx].squeeze(-1)

            if rescale:
                scale_factor = img_metas[idx]['scale_factor']
                bboxes_per_img /= bboxes_per_img.new_tensor(scale_factor)

            if not self.use_nms:
                bboxes_per_img = torch.cat([bboxes_per_img, all_scores[idx]], dim=1)
                result_list.append((bboxes_per_img, labels_per_img))
            else:
                labels_per_img += 1
                bboxes_per_img, labels_per_img = nms_agnostic(
                    bboxes_per_img, scores_per_img.squeeze(-1), labels_per_img,
                    cfg.score_thr, cfg.nms, self.num_classes + 1, cfg.max_per_img)
                result_list.append((bboxes_per_img, labels_per_img))

        return result_list

    def loss(self,
             pred_heatmap,
             pred_wh,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        all_targets = self._target_generator(gt_bboxes, gt_labels, img_metas)
        hm_loss, wh_loss = self._loss(pred_heatmap, pred_wh, *all_targets)
        return {'losses/centernet_loss_heatmap': hm_loss,
                'losses/centernet_loss_wh': wh_loss}

    def train(self, mode=True):
        super(MLCTHead, self).train(mode)
        if mode and self.norm_eval:
            for _, m in self.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

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


class MLCTTargetGenerator(object):

    def __init__(self,
                 num_classes,
                 fpn_strides,
                 obj_sizes_of_interest,
                 use_giou,
                 max_objs=100):
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.obj_sizes_of_interest = obj_sizes_of_interest
        self.use_giou = use_giou
        self.max_objs = max_objs

    def target_single_image(self, gt_boxes, gt_labels, feat_shapes, obj_sizes_of_interest):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: list(tuple). tuple <=> level.
            obj_sizes_of_interest: tensor, (level_num, 2).

        Returns:
            all_heatmap: tensor, tensor <=> img, (80, h*w for all levels).
            all_wh: tensor, tensor <=> img, (max_obj*level_num, 2).
            all_reg_mask: tensor, tensor <=> img, (max_obj*level_num,).
            all_ind: tensor, tensor <=> img, (max_obj*level_num,).
            all_center_location: tensor or None, tensor <=> img, (max_obj*level_num, 2).
        """
        level_size = len(self.fpn_strides)
        all_heatmap, all_wh, all_reg_mask, all_ind, all_center_location = [], [], [], [], []
        max_wh_target = torch.max(
            gt_boxes[:, 3] - gt_boxes[:, 1],
            gt_boxes[:, 2] - gt_boxes[:, 0]).unsqueeze(-1).repeat(1, level_size)
        is_cared_in_the_level = \
            (max_wh_target >= obj_sizes_of_interest[:, 0]) & \
            (max_wh_target <= obj_sizes_of_interest[:, 1])  # (gt_num, level_num)

        cared_gt_num_per_level = {}
        for lvl in range(level_size):
            cared_gt_num_per_level['num_lv{}'.format(lvl)] = \
                is_cared_in_the_level[:, lvl].sum().item()
        add_summary('centernet', **cared_gt_num_per_level)

        for lvl in range(level_size):
            # get target for a single level of a single image.
            output_h, output_w = feat_shapes[lvl]
            heatmap = gt_boxes.new_zeros((self.num_classes, output_h, output_w))
            wh = gt_boxes.new_zeros((self.max_objs, 2))
            reg_mask = gt_boxes.new_zeros((self.max_objs,), dtype=torch.uint8)
            ind = gt_boxes.new_zeros((self.max_objs,), dtype=torch.long)

            center_location = None
            if self.use_giou:
                center_location = gt_boxes.new_zeros((self.max_objs, 2))

            gt_boxes_in_lvl = gt_boxes[is_cared_in_the_level[:, lvl]]
            if gt_boxes_in_lvl.size(0) > 0:
                gt_boxes_in_lvl /= self.fpn_strides[lvl]
                gt_boxes_in_lvl[:, [0, 2]] = torch.clamp(gt_boxes_in_lvl[:, [0, 2]],
                                                         0, output_w - 1)
                gt_boxes_in_lvl[:, [1, 3]] = torch.clamp(gt_boxes_in_lvl[:, [1, 3]],
                                                         0, output_h - 1)
                hs = gt_boxes_in_lvl[:, 3] - gt_boxes_in_lvl[:, 1]
                ws = gt_boxes_in_lvl[:, 2] - gt_boxes_in_lvl[:, 0]

                if every_n_local_step(500):
                    add_histogram_summary('mlct_head_hs_lv{}'.format(lvl),
                                          hs.detach().cpu(), collect_type='none')
                    add_histogram_summary('mlct_head_ws_lv{}'.format(lvl),
                                          ws.detach().cpu(), collect_type='none')

                for k in range(gt_boxes_in_lvl.shape[0]):
                    cls_id = gt_labels[k] - 1
                    h, w = hs[k], ws[k]
                    if h > 0 and w > 0:
                        radius = gaussian_radius((h.ceil(), w.ceil()))
                        radius = max(0, int(radius.item()))
                        center = gt_boxes.new_tensor([
                            (gt_boxes_in_lvl[k, 0] + gt_boxes_in_lvl[k, 2]) / 2,
                            (gt_boxes_in_lvl[k, 1] + gt_boxes_in_lvl[k, 3]) / 2])
                        # no peak will fall between pixels
                        ct_int = center.to(torch.int)
                        draw_umich_gaussian(heatmap[cls_id], ct_int, radius)
                        # directly predict the width and height
                        wh[k] = wh.new_tensor([1. * w, 1. * h])
                        ind[k] = ct_int[1] * output_w + ct_int[0]
                        if self.use_giou:
                            center_location[k] = center
                        reg_mask[k] = 1

            all_heatmap.append(heatmap.view(heatmap.shape[0], -1))
            all_wh.append(wh)
            all_reg_mask.append(reg_mask)
            all_ind.append(ind)
            all_center_location.append(center_location)

        all_heatmap, all_reg_mask, all_ind = [torch.cat(x, dim=-1)
                                              for x in [all_heatmap, all_reg_mask, all_ind]]
        all_wh = torch.cat(all_wh, dim=0)
        if self.use_giou:
            all_center_location = torch.cat(all_center_location, dim=0)

        return all_heatmap, all_wh, all_reg_mask, all_ind, all_center_location

    def __call__(self, gt_boxes, gt_labels, img_metas):
        """

        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:
            heatmap: tensor, (batch, 80, h*w for all levels).
            wh: tensor, (batch, max_obj*level_num, 2).
            reg_mask: tensor, tensor <=> img, (batch, max_obj*level_num).
            ind: tensor, (batch, max_obj*level_num).
            center_location: tensor or None, (batch, max_obj*level_num, 2).
        """
        with torch.no_grad():
            pad_h, pad_w = img_metas[0]['pad_shape'][0:2]
            feat_shapes = [(pad_h // down_ratio, pad_w // down_ratio)
                           for down_ratio in self.fpn_strides]
            obj_sizes_of_interest = gt_boxes[0].new_tensor(self.obj_sizes_of_interest)

            heatmap, wh, reg_mask, ind, center_location = multi_apply(
                self.target_single_image,
                gt_boxes,
                gt_labels,
                feat_shapes=feat_shapes,
                obj_sizes_of_interest=obj_sizes_of_interest
            )

            heatmap, wh, ind, reg_mask = [torch.stack(t, dim=0).detach()
                                          for t in [heatmap, wh, ind, reg_mask]]
            if self.use_giou:
                center_location = torch.stack(center_location, dim=0).detach()

            return heatmap, wh, reg_mask, ind, center_location


class MLCTLoss(object):

    def __init__(self,
                 use_smooth_l1=False,
                 use_giou=False,
                 wh_weight=0.1,
                 hm_weight=1.,
                 gamma=2.):
        self.use_smooth_l1 = use_smooth_l1
        self.use_giou = use_giou
        self.wh_weight = wh_weight
        self.hm_weight = hm_weight
        self.gamma = gamma

    def __call__(self,
                 pred_hm,
                 pred_wh,
                 heatmap,
                 wh,
                 reg_mask,
                 ind,
                 center_location):
        """

        Args:
            pred_hm: list(tensor), tensor <=> batch, (batch, 80, h, w).
            pred_wh: list(tensor), tensor <=> batch, (batch, 2, h, w).
            heatmap: tensor, (batch, 80, h*w for all levels).
            wh: tensor, (batch, max_obj*level_num, 2).
            reg_mask: tensor, tensor <=> img, (batch, max_obj*level_num).
            ind: tensor, (batch, max_obj*level_num).
            center_location: tensor or None, (batch, max_obj*level_num, 2). Only useful when
                using GIOU.

        Returns:

        """
        if every_n_local_step(500):
            for lvl, hm in enumerate(pred_hm):
                hm_summary = hm.clone().detach().sigmoid_()
                add_feature_summary('centernet_heatmap_lv{}'.format(lvl),
                                    hm_summary.cpu().numpy())

        H, W = pred_hm[0].shape[2:]
        level_num = len(pred_hm)
        pred_hm = torch.cat([x.view(*x.shape[:2], -1) for x in pred_hm], dim=-1)
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        hm_loss = ct_focal_loss(pred_hm, heatmap, self.gamma) * self.hm_weight

        # (batch, 2, h, w) for all levels => (batch, max_obj*level_num, 2)
        ind_levels = ind.chunk(level_num, dim=1)
        pred_wh_pruned = []
        for pred_wh_per_lvl, ind_lvl in zip(pred_wh, ind_levels):
            pred_wh_pruned.append(tranpose_and_gather_feat(pred_wh_per_lvl, ind_lvl))
        pred_wh_pruned = torch.cat(pred_wh_pruned, dim=1)  # (batch, max_obj*level_num, 2)
        mask = reg_mask.unsqueeze(2).expand_as(pred_wh_pruned).float()
        avg_factor = mask.sum() + 1e-4

        if self.use_giou:
            pred_boxes = torch.cat(
                (center_location - pred_wh_pruned / 2.,
                 center_location + pred_wh_pruned / 2.), dim=2)
            box_br = center_location + wh / 2.
            box_br[:, :, 0] = box_br[:, :, 0].clamp(max=W - 1)
            box_br[:, :, 1] = box_br[:, :, 1].clamp(max=H - 1)
            box_tl = torch.clamp(center_location - wh / 2., min=0)
            boxes = torch.cat((box_tl, box_br), dim=2)
            mask_expand_4 = mask.repeat(1, 1, 2)
            wh_loss = giou_loss(pred_boxes, boxes, mask_expand_4)
        else:
            if self.use_smooth_l1:
                wh_loss = smooth_l1_loss(pred_wh_pruned, wh, mask,
                                         avg_factor=avg_factor) * self.wh_weight
            else:
                wh_loss = weighted_l1(pred_wh_pruned, wh, mask,
                                      avg_factor=avg_factor) * self.wh_weight

        return hm_loss, wh_loss
