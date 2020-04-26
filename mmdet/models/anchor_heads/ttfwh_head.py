import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init, constant_init
import math
import numpy as np

from mmdet.ops import ModulatedDeformConvPack, soft_nms
from mmdet.core import multi_apply, force_fp32
from mmdet.models.losses import ct_focal_loss, giou_loss
from mmdet.models.utils import (
    build_conv_layer, build_norm_layer, bias_init_with_prob, ConvModule)
from ..registry import HEADS
from .anchor_head import AnchorHead


class UpsamplingLayers(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='BN'), no_upsample=False):
        mdcn = ModulatedDeformConvPack(
            in_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=1)
        layers = []
        layers.append(mdcn)
        if norm_cfg:
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        layers.append(nn.ReLU(inplace=True))
        if not no_upsample:
            layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
        super(UpsamplingLayers, self).__init__(*layers)


class ShortcutConnection(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, conv_cfg):
        super(ShortcutConnection, self).__init__()
        layers = []
        for i, kernel_size in enumerate(kernel_sizes):
            inc = in_channels if i == 0 else out_channels
            padding = (kernel_size - 1) // 2
            if conv_cfg:
                layers.append(
                    build_conv_layer(conv_cfg, inc, out_channels, kernel_size,
                                     padding=padding))
            else:
                layers.append(
                    nn.Conv2d(inc, out_channels, kernel_size, padding=padding))
            if i < len(kernel_sizes) - 1:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


@HEADS.register_module
class TTFLevelHead(AnchorHead):

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 down_ratio_b1=8,
                 down_ratio_b2=4,
                 hm_head_channels=256,
                 wh_head_channels=(64, 64),
                 hm_head_conv_num=2,
                 wh_head_conv_num=(2, 2),
                 num_classes=81,
                 shortcut_cfg=(1, 2, 3),
                 wh_scale_factor_b1=16.,
                 wh_scale_factor_b2=16.,
                 alpha=0.54,
                 beta=0.54,
                 hm_weight=1.,
                 wh_weight_b1=5.,
                 wh_weight_b2=5.,
                 b1_min_length=32,
                 b2_max_length=64,
                 level_base_area=True,
                 inf_branch=['b1', 'b2'],
                 use_simple_nms=True,
                 focal_loss_beta=4,
                 focal_b2_only=False,
                 shortcut_conv_cfg=None,
                 head_conv_cfg=None,
                 inf_branch_filter=False,
                 max_objs=128,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(AnchorHead, self).__init__()
        assert len(inplanes) == 4 and len(planes) == 3 and len(shortcut_cfg) == 3
        self.inplanes = inplanes
        self.planes = planes
        self.down_ratio_b1 = down_ratio_b1
        self.down_ratio_b2 = down_ratio_b2
        self.hm_head_channels = hm_head_channels
        self.wh_head_channels = wh_head_channels
        self.hm_head_conv_num = hm_head_conv_num
        self.wh_head_conv_num = wh_head_conv_num
        self.num_classes = num_classes
        self.num_fg = num_classes - 1
        self.shortcut_cfg = shortcut_cfg
        self.wh_scale_factor_b1 = wh_scale_factor_b1
        self.wh_scale_factor_b2 = wh_scale_factor_b2
        self.alpha = alpha
        self.beta = beta
        self.max_objs = max_objs
        self.wh_weight_b1 = wh_weight_b1
        self.wh_weight_b2 = wh_weight_b2
        self.b1_min_length = b1_min_length
        self.b2_max_length = b2_max_length
        self.level_base_area = level_base_area
        self.inf_branch = inf_branch
        self.use_simple_nms = use_simple_nms
        self.focal_loss_beta = focal_loss_beta
        self.focal_b2_only = focal_b2_only
        self.shortcut_conv_cfg = shortcut_conv_cfg
        self.head_conv_cfg = head_conv_cfg
        self.inf_branch_filter = inf_branch_filter
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.base_loc_b1 = None
        self.base_loc_b2 = None
        self.fp16_enabled = False

        self._init_layers()

    def _init_branch_layers(self, planes):
        wh_layers, wh2_layers, hm_layers = [], [], []
        inp = planes
        for i in range(self.wh_head_conv_num[0]):
            wh_layers.append(
                ConvModule(
                    inp,
                    self.wh_head_channels[0],
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg))
            inp = self.wh_head_channels[0]
        if self.head_conv_cfg:
            wh_layers.append(
                build_conv_layer(
                    self.head_conv_cfg,
                    self.wh_head_channels[0],
                    4,
                    kernel_size=3,
                    padding=1
                )
            )
        else:
            wh_layers.append(nn.Conv2d(self.wh_head_channels[0], 4, 3, padding=1))

        inp = planes
        for i in range(self.wh_head_conv_num[1]):
            wh2_layers.append(
                ConvModule(
                    inp,
                    self.wh_head_channels[1],
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg))
            inp = self.wh_head_channels[1]
        if self.head_conv_cfg:
            wh2_layers.append(
                build_conv_layer(
                    self.head_conv_cfg,
                    self.wh_head_channels[1],
                    4,
                    kernel_size=3,
                    padding=1
                )
            )
        else:
            wh2_layers.append(nn.Conv2d(self.wh_head_channels[1], 4, 3, padding=1))

        inp = planes
        for i in range(self.hm_head_conv_num):
            hm_layers.append(
                ConvModule(
                    inp,
                    self.hm_head_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg))
            inp = self.hm_head_channels
        if self.head_conv_cfg:
            hm_layers.append(
                build_conv_layer(
                    self.head_conv_cfg,
                    self.hm_head_channels,
                    self.num_fg,
                    kernel_size=3,
                    padding=1
                )
            )
        else:
            hm_layers.append(nn.Conv2d(self.hm_head_channels, self.num_fg, 3, padding=1))

        wh_layers = nn.Sequential(*wh_layers)
        wh2_layers = nn.Sequential(*wh2_layers)
        hm_layers = nn.Sequential(*hm_layers)
        return wh_layers, wh2_layers, hm_layers

    def _init_layers(self):
        self.upsample_layers = nn.ModuleList([
            UpsamplingLayers(
                self.inplanes[-1], self.planes[0], norm_cfg=self.norm_cfg),
            UpsamplingLayers(
                self.planes[0], self.planes[1], norm_cfg=self.norm_cfg),
            UpsamplingLayers(
                self.planes[1], self.planes[2], norm_cfg=self.norm_cfg)
        ])

        self.shortcut_layers = nn.ModuleList()
        for (inp, outp, layer_num) in zip(self.inplanes[::-1][1:],
                                          self.planes, self.shortcut_cfg):
            assert layer_num > 0, "Shortcut connection must be included."
            self.shortcut_layers.append(
                ShortcutConnection(inp, outp, [3] * layer_num, self.shortcut_conv_cfg))

        self.wh_b1, self.wh_b2, self.hm = self._init_branch_layers(self.planes[-1])

    def init_weights(self):
        for m in self.upsample_layers.modules():
            if isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

        for m in self.shortcut_layers.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

        bias_cls = bias_init_with_prob(0.01)
        for m in self.hm.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        normal_init(self.hm[-1], std=0.01, bias=bias_cls)

        for wh in [self.wh_b1, self.wh_b2]:
            for m in wh.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

        for m in self.modules():
            if isinstance(m, ModulatedDeformConvPack):
                constant_init(m.conv_offset, 0)

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: list(tensor), (batch, 80, h, w).
            wh: list(tensor), (batch, 4, h, w).
        """
        y, shortcuts = [], []
        x = feats[-1]
        for i, shortcut_layer in enumerate(self.shortcut_layers):
            shortcuts.append(shortcut_layer(feats[-i - 2]))

        for i, upsampling_layer in enumerate(self.upsample_layers):
            x = upsampling_layer(x)
            x = x + shortcuts[i]
            y.append(x)

        y_s4 = y[-1]
        hm = self.hm(y_s4)
        wh_b1 = F.relu(self.wh_b1(y_s4)) * self.wh_scale_factor_b1
        wh_b2 = F.relu(self.wh_b2(y_s4)) * self.wh_scale_factor_b2

        return hm, wh_b1, wh_b2

    def get_bboxes_single(self,
                          pred_hm,
                          pred_wh,
                          down_ratio,
                          topk,
                          idx=0):
        batch, cat, height, width = pred_hm.size()
        pred_hm = pred_hm.detach().sigmoid_()
        wh = pred_wh.detach()

        # used maxpool to filter the max score
        heat = self.simple_nms(pred_hm)

        # (batch, topk)
        scores, inds, clses, ys, xs = self._topk(heat, topk=topk)
        xs = xs.view(batch, topk, 1) * down_ratio
        ys = ys.view(batch, topk, 1) * down_ratio

        wh = wh.permute(0, 2, 3, 1).contiguous()
        wh = wh.view(wh.size(0), -1, wh.size(3))
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), wh.size(2))
        wh = wh.gather(1, inds)

        wh_filter = wh.new_ones((batch, topk), dtype=torch.bool)
        if self.inf_branch_filter:
            area = (wh[..., 2] + wh[..., 0] + 1) * (wh[..., 3] + wh[..., 1] + 1)
            if idx == 0:
                wh_filter = area >= self.b1_min_length ** 2 / 2
            elif idx == 1:
                wh_filter = area <= self.b2_max_length ** 2 * 2

        wh = wh.view(batch, topk, 4)
        clses = clses.view(batch, topk, 1).float()
        scores = scores.view(batch, topk, 1)

        bboxes = torch.cat([
            xs - wh[..., [0]], ys - wh[..., [1]], xs + wh[..., [2]],
            ys + wh[..., [3]]
        ], dim=2)
        return heat, inds, clses, scores, bboxes, xs, ys, wh_filter

    @force_fp32(apply_to=('pred_hm_b1', 'pred_hm_b2', 'pred_wh_b1', 'pred_wh_b2'))
    def get_bboxes(self,
                   pred_hm,
                   pred_wh_b1,
                   pred_wh_b2,
                   img_metas,
                   cfg,
                   rescale=False):
        topk = getattr(cfg, 'max_per_img', 100)
        heat_b1, inds_b1, clses_b1, scores_b1, bboxes_b1, xs_b1, ys_b1, wh_filter_b1 = \
            self.get_bboxes_single(pred_hm_b1, pred_wh_b1, self.down_ratio_b1, topk, idx=0)
        heat_b2, inds_b2, clses_b2, scores_b2, bboxes_b2, xs_b2, ys_b2, wh_filter_b2 = \
            self.get_bboxes_single(pred_hm_b2, pred_wh_b2, self.down_ratio_b2, topk, idx=1)

        # heat_b2 = nn.functional.max_pool2d(heat_b2, (3, 3), stride=1, padding=1)
        # heat_b2 = nn.functional.max_pool2d(heat_b2, (2, 2), stride=2)
        # collide_heat = ((heat_b1 > 0) & (heat_b2 > 0)).max(1)[0]
        # collide_heat = collide_heat.view(collide_heat.size(0), -1)  # (batch, h * w)
        # collide_b1 = collide_heat.gather(-1, )

        result_list = []
        score_thr = getattr(cfg, 'score_thr', 0.01)
        if 'b2' not in self.inf_branch:
            bboxes = bboxes_b1
            scores = scores_b1
            clses = clses_b1
            wh_filter = wh_filter_b1
        elif 'b1' not in self.inf_branch:
            bboxes = bboxes_b2
            scores = scores_b2
            clses = clses_b2
            wh_filter = wh_filter_b2
        else:
            bboxes = torch.cat([bboxes_b1, bboxes_b2], dim=1)
            scores = torch.cat([scores_b1, scores_b2], dim=1)
            clses = torch.cat([clses_b1, clses_b2], dim=1)
            wh_filter = torch.cat([wh_filter_b1, wh_filter_b2], dim=1)

        for batch_i in range(bboxes.shape[0]):
            scores_per_img = scores[batch_i]
            wh_filter_per_img = wh_filter[batch_i]
            scores_keep = (scores_per_img > score_thr).squeeze(-1) & wh_filter_per_img

            scores_per_img = scores_per_img[scores_keep]
            bboxes_per_img = bboxes[batch_i][scores_keep]
            labels_per_img = clses[batch_i][scores_keep].squeeze(-1)
            img_shape = img_metas[batch_i]['pad_shape']
            bboxes_per_img[:, 0::2] = bboxes_per_img[:, 0::2].clamp(
                min=0, max=img_shape[1] - 1)
            bboxes_per_img[:, 1::2] = bboxes_per_img[:, 1::2].clamp(
                min=0, max=img_shape[0] - 1)

            if rescale:
                scale_factor = img_metas[batch_i]['scale_factor']
                bboxes_per_img /= bboxes_per_img.new_tensor(scale_factor)

            if self.use_simple_nms:
                bboxes_per_img = torch.cat([bboxes_per_img, scores_per_img], dim=1)
            else:
                labels_int_flatten = labels_per_img.int()
                unique_cls_ids = list(set(list(labels_int_flatten.cpu().numpy())))
                bboxes_per_img_per_cls = bboxes_per_img.new_zeros((0, 5))
                labels_per_img_per_cls = labels_int_flatten.new_zeros((0,))
                for cls_id in unique_cls_ids:
                    cls_id_idx = (labels_int_flatten == cls_id)
                    soft_bboxes, ori_idx = soft_nms(torch.cat((
                        bboxes_per_img[cls_id_idx], scores_per_img[cls_id_idx]), dim=1),
                        iou_thr=0.6)
                    unique_labels = labels_int_flatten[cls_id_idx][ori_idx]
                    bboxes_per_img_per_cls = torch.cat((bboxes_per_img_per_cls, soft_bboxes), dim=0)
                    labels_per_img_per_cls = torch.cat((labels_per_img_per_cls, unique_labels))
                bboxes_per_img = bboxes_per_img_per_cls
                labels_per_img = labels_per_img_per_cls.float()

            labels_per_img = labels_per_img.squeeze(-1)
            result_list.append((bboxes_per_img, labels_per_img))

        return result_list

    def loss_single(self,
                    pred_hm,
                    pred_wh,
                    heatmap,
                    box_target,
                    wh_weight,
                    down_ratio,
                    base_loc_name,
                    hm_weight_factor,
                    wh_weight_factor,
                    focal_loss_beta):
        H, W = pred_hm.shape[2:]
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        loss_cls = ct_focal_loss(pred_hm, heatmap,
                                 beta=focal_loss_beta) * hm_weight_factor

        if getattr(self, base_loc_name) is None or H != getattr(self, base_loc_name).shape[
            1] or W != getattr(self, base_loc_name).shape[2]:
            base_step = down_ratio
            shifts_x = torch.arange(
                0, (W - 1) * base_step + 1,
                base_step,
                dtype=torch.float32,
                device=heatmap.device)
            shifts_y = torch.arange(
                0, (H - 1) * base_step + 1,
                base_step,
                dtype=torch.float32,
                device=heatmap.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            setattr(self, base_loc_name, torch.stack((shift_x, shift_y), dim=0))  # (2, h, w)

        # (batch, h, w, 4)
        pred_boxes = torch.cat((getattr(self, base_loc_name) - pred_wh[:, [0, 1]],
                                getattr(self, base_loc_name) + pred_wh[:, [2, 3]]),
                               dim=1).permute(0, 2, 3, 1)
        boxes = box_target.permute(0, 2, 3, 1)

        mask = wh_weight.view(-1, H, W)
        avg_factor = mask.sum() + 1e-4

        loss_bbox = giou_loss(
            pred_boxes, boxes, mask, avg_factor=avg_factor) * wh_weight_factor

        return loss_cls, loss_bbox

    @force_fp32(apply_to=('pred_hm_b1', 'pred_hm_b2', 'pred_wh_b1', 'pred_wh_b2'))
    def loss(self,
             pred_hm_b1,
             pred_hm_b2,
             pred_wh_b1,
             pred_wh_b2,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        h_b1, h_b2, b_b1, b_b2, r_b1, r_b2 = self.ttf_target(
            gt_bboxes, gt_labels, img_metas)

        loss_cls_b1, loss_bbox_b1 = self.loss_single(
            pred_hm_b1, pred_wh_b1, h_b1, b_b1, r_b1,
            self.down_ratio_b1, 'base_loc_b1', self.hm_weight_b1, self.wh_weight_b1,
            4 if self.focal_b2_only else self.focal_loss_beta)
        loss_cls_b2, loss_bbox_b2 = self.loss_single(
            pred_hm_b2, pred_wh_b2, h_b2, b_b2, r_b2,
            self.down_ratio_b2, 'base_loc_b2', self.hm_weight_b2, self.wh_weight_b2,
            self.focal_loss_beta)

        return {'losses/ttfnetv2_loss_hm_b1': loss_cls_b1,
                'losses/ttfnetv2_loss_wh_b1': loss_bbox_b1,
                'losses/ttfnetv2_loss_hm_b2': loss_cls_b2,
                'losses/ttfnetv2_loss_wh_b2': loss_bbox_b2}

    def _topk(self, scores, topk):
        batch, cat, height, width = scores.size()

        # (batch, 80, topk)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), topk)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # (batch, topk). select topk from 80*topk
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
        topk_clses = (topk_ind / topk).int()
        topk_ind = topk_ind.unsqueeze(2)
        topk_inds = topk_inds.view(batch, -1,
                                   1).gather(1, topk_ind).view(batch, topk)
        topk_ys = topk_ys.view(batch, -1,
                               1).gather(1, topk_ind).view(batch, topk)
        topk_xs = topk_xs.view(batch, -1,
                               1).gather(1, topk_ind).view(batch, topk)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y /
                     (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = heatmap.new_tensor(gaussian)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius -
                                                                     left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def ttf_target_single_single(self,
                                 heatmap,
                                 box_target,
                                 reg_weight,
                                 fake_heatmap,
                                 boxes_area_topk_log,
                                 gt_boxes,
                                 gt_labels,
                                 boxes_ind,
                                 feat_shape,
                                 down_ratio):
        output_h, output_w = feat_shape
        feat_gt_boxes = gt_boxes / down_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(
            feat_gt_boxes[:, [0, 2]], min=0, max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(
            feat_gt_boxes[:, [1, 3]], min=0, max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / down_ratio).to(torch.int)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
        if self.alpha != self.beta:
            h_radiuses_beta = (feat_hs / 2. * self.beta).int()
            w_radiuses_beta = (feat_ws / 2. * self.beta).int()

        # larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k] - 1

            fake_heatmap = fake_heatmap.zero_()
            self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item(),
                                        w_radiuses_alpha[k].item())
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)

            if self.alpha != self.beta:
                fake_heatmap = fake_heatmap.zero_()
                self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                            h_radiuses_beta[k].item(),
                                            w_radiuses_beta[k].item())
            box_target_inds = fake_heatmap > 0

            box_target[:, box_target_inds] = gt_boxes[k][:, None]
            cls_id = 0

            local_heatmap = fake_heatmap[box_target_inds]
            ct_div = local_heatmap.sum()
            local_heatmap *= boxes_area_topk_log[k]
            reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
        return heatmap, box_target, reg_weight

    def ttf_target_single(self, gt_boxes, gt_labels, feat_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w).
            reg_weight: tensor, same as box_target
        """
        output_h_b1, output_w_b1, output_h_b2, output_w_b2 = feat_shape
        heatmap_channel = self.num_fg

        heatmap_b1 = gt_boxes.new_zeros((heatmap_channel, output_h_b1, output_w_b1))
        fake_heatmap_b1 = gt_boxes.new_zeros((output_h_b1, output_w_b1))
        box_target_b1 = gt_boxes.new_ones((4, output_h_b1, output_w_b1)) * -1
        reg_weight_b1 = gt_boxes.new_zeros((1, output_h_b1, output_w_b1))
        heatmap_b2 = gt_boxes.new_zeros((heatmap_channel, output_h_b2, output_w_b2))
        fake_heatmap_b2 = gt_boxes.new_zeros((output_h_b2, output_w_b2))
        box_target_b2 = gt_boxes.new_ones((4, output_h_b2, output_w_b2)) * -1
        reg_weight_b2 = gt_boxes.new_zeros((1, output_h_b2, output_w_b2))

        boxes_areas_log = self.bbox_areas(gt_boxes).log()
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log,
                                                    boxes_areas_log.size(0))

        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        if self.level_base_area:
            gt_b1_idx = boxes_area_topk_log >= math.log(self.b1_min_length ** 2)
            gt_b2_idx = boxes_area_topk_log <= math.log(self.b2_max_length ** 2)
        else:
            gt_b1_idx = gt_boxes.max(-1)[0] >= self.b1_min_length
            gt_b2_idx = gt_boxes.max(-1)[0] <= self.b2_max_length
        heatmap_b1, box_target_b1, reg_weight_b1 = self.ttf_target_single_single(
            heatmap_b1,
            box_target_b1,
            reg_weight_b1,
            fake_heatmap_b1,
            boxes_area_topk_log[gt_b1_idx],
            gt_boxes[gt_b1_idx],
            gt_labels[gt_b1_idx],
            boxes_ind[gt_b1_idx],
            [output_h_b1, output_w_b1],
            self.down_ratio_b1)

        heatmap_b2, box_target_b2, reg_weight_b2 = self.ttf_target_single_single(
            heatmap_b2,
            box_target_b2,
            reg_weight_b2,
            fake_heatmap_b2,
            boxes_area_topk_log[gt_b2_idx],
            gt_boxes[gt_b2_idx],
            gt_labels[gt_b2_idx],
            boxes_ind[gt_b2_idx],
            [output_h_b2, output_w_b2],
            self.down_ratio_b2)

        return heatmap_b1, heatmap_b2, box_target_b1, box_target_b2, reg_weight_b1, reg_weight_b2

    def ttf_target(self, gt_boxes, gt_labels, img_metas):
        """

        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w).
            reg_weight: tensor, (batch, 1, h, w).
        """
        with torch.no_grad():
            feat_shape = (img_metas[0]['pad_shape'][0] // self.down_ratio_b1,
                          img_metas[0]['pad_shape'][1] // self.down_ratio_b1,
                          img_metas[0]['pad_shape'][0] // self.down_ratio_b2,
                          img_metas[0]['pad_shape'][1] // self.down_ratio_b2)

            h_b1, h_b2, b_b1, b_b2, r_b1, r_b2 = multi_apply(
                self.ttf_target_single,
                gt_boxes,
                gt_labels,
                feat_shape=feat_shape)

            h_b1, h_b2, b_b1, b_b2, r_b1, r_b2 = [
                torch.stack(t, dim=0).detach()
                for t in [h_b1, h_b2, b_b1, b_b2, r_b1, r_b2]
            ]

            return h_b1, h_b2, b_b1, b_b2, r_b1, r_b2

    def simple_nms(self, heat, kernel=3, out_heat=None):
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        out_heat = heat if out_heat is None else out_heat
        return out_heat * keep

    def bbox_areas(self, bboxes, keep_axis=False):
        x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], \
                                     bboxes[:, 2], bboxes[:, 3]
        areas = (y_max - y_min + 1) * (x_max - x_min + 1)
        if keep_axis:
            return areas[:, None]
        return areas
