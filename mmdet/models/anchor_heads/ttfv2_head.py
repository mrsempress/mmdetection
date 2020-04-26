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

    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='BN')):
        mdcn = ModulatedDeformConvPack(
            in_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=1)
        up = nn.UpsamplingBilinear2d(scale_factor=2)

        layers = []
        layers.append(mdcn)
        if norm_cfg:
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        layers.append(nn.ReLU(inplace=True))
        layers.append(up)
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


class DepthwiseHead(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes,
                 conv_relu=True, depth_kernel_sizes=None, depth_group=None,
                 use_deform=False):
        super(DepthwiseHead, self).__init__()
        self.conv_relu = conv_relu
        if depth_kernel_sizes is None:
            depth_kernel_sizes = kernel_sizes
        if depth_group is None:
            depth_group = out_channels

        padding = (kernel_sizes - 1) // 2
        depth_padding = (depth_kernel_sizes - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_sizes, padding=padding)
        if use_deform:
            self.depth_conv = ModulatedDeformConvPack(
                out_channels, out_channels, depth_kernel_sizes,
                padding=depth_padding, groups=depth_group)
        else:
            self.depth_conv = nn.Conv2d(out_channels, out_channels, depth_kernel_sizes,
                                        padding=depth_padding, groups=depth_group)

    def forward(self, x):
        y = self.conv(x)
        z = F.relu(y) if self.conv_relu else y
        return y + self.depth_conv(z)


@HEADS.register_module
class TTFv2Head(AnchorHead):

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes_b1=(128, 64),
                 planes_b2=(64, 32),
                 down_ratio_b1=8,
                 down_ratio_b2=4,
                 hm_head_channels=256,
                 wh_head_channels=64,
                 hm_head_conv_num=2,
                 wh_head_conv_num=2,
                 num_classes=81,
                 shortcut_cfg=(1, 2),
                 wh_scale_factor_b1=16.,
                 wh_scale_factor_b2=16.,
                 alpha=0.54,
                 beta=0.54,
                 hm_weight_b1=1.,
                 wh_weight_b1=5.,
                 hm_weight_b2=1.,
                 wh_weight_b2=5.,
                 b1_min_length=32,
                 b2_max_length=64,
                 level_base_area=True,
                 inf_branch=['b1', 'b2'],
                 use_simple_nms=True,
                 get_bboxesv2=False,
                 b1_s16_conv_num=0,
                 b1_s16_bn=False,
                 b1_s8_conv_num=0,
                 b1_s8_bn=False,
                 b1_last_relu=False,
                 b1_concat=False,
                 b1_minus=False,
                 use_b2_extra_layers=False,
                 extra_layer_num=1,
                 extra_layer_bn=False,
                 norm_log=True,
                 focal_loss_beta=4,
                 focal_b2_only=False,
                 shortcut_conv_cfg=None,
                 head_conv_cfg=None,
                 depthwise_hm=False,
                 relu_before_depthwise=True,
                 depth_kernel_sizes=3,
                 depth_group=None,
                 depth_init_kaiming=False,
                 depth_deform=False,
                 max_objs=128,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(AnchorHead, self).__init__()
        assert len(inplanes) == 4 and len(planes_b1) == 2 and len(planes_b2) == 2 and len(
            shortcut_cfg) == 2
        self.inplanes = inplanes
        self.planes_b1 = planes_b1
        self.planes_b2 = planes_b2
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
        self.hm_weight_b1 = hm_weight_b1
        self.wh_weight_b1 = wh_weight_b1
        self.hm_weight_b2 = hm_weight_b2
        self.wh_weight_b2 = wh_weight_b2
        self.b1_min_length = b1_min_length
        self.b2_max_length = b2_max_length
        self.level_base_area = level_base_area
        self.inf_branch = inf_branch
        self.use_simple_nms = use_simple_nms
        self.get_bboxesv2 = get_bboxesv2
        self.b1_s16_conv_num = b1_s16_conv_num
        self.b1_s16_bn = b1_s16_bn
        self.b1_s8_conv_num = b1_s8_conv_num
        self.b1_s8_bn = b1_s8_bn
        self.b1_last_relu = b1_last_relu
        self.b1_concat = b1_concat
        self.b1_minus = b1_minus
        self.use_b2_extra_layers = use_b2_extra_layers
        self.extra_layer_num = extra_layer_num
        self.extra_layer_bn = extra_layer_bn
        self.norm_log = norm_log
        self.focal_loss_beta = focal_loss_beta
        self.focal_b2_only = focal_b2_only
        self.shortcut_conv_cfg = shortcut_conv_cfg
        self.head_conv_cfg = head_conv_cfg
        self.depthwise_hm = depthwise_hm
        self.relu_before_depthwise = relu_before_depthwise
        self.depth_kernel_sizes = depth_kernel_sizes
        self.depth_group = depth_group
        self.depth_init_kaiming = depth_init_kaiming
        self.depth_deform = depth_deform
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.base_loc_b1 = None
        self.base_loc_b2 = None
        self.fp16_enabled = False

        self._init_layers()

    def _init_branch_layers(self, planes, upsample_inplane, shortcut_inplanes):
        upsample_layers = nn.ModuleList([
            UpsamplingLayers(
                upsample_inplane, planes[0], norm_cfg=self.norm_cfg),
            UpsamplingLayers(
                planes[0], planes[1], norm_cfg=self.norm_cfg),
        ])

        shortcut_layers = nn.ModuleList()
        for (inp, outp, layer_num) in zip(shortcut_inplanes,
                                          planes, self.shortcut_cfg):
            assert layer_num > 0, "Shortcut connection must be included."
            shortcut_layers.append(
                ShortcutConnection(inp, outp, [3] * layer_num, self.shortcut_conv_cfg))

        wh_layers, hm_layers = [], []
        inp = planes[-1]
        for i in range(self.wh_head_conv_num):
            wh_layers.append(
                ConvModule(
                    inp,
                    self.wh_head_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg))
            inp = self.wh_head_channels
        if self.head_conv_cfg:
            wh_layers.append(
                build_conv_layer(
                    self.head_conv_cfg,
                    self.wh_head_channels,
                    4,
                    kernel_size=3,
                    padding=1
                )
            )
        else:
            wh_layers.append(nn.Conv2d(self.wh_head_channels, 4, 3, padding=1))

        inp = planes[-1]
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
        elif self.depthwise_hm:
            hm_layers.append(DepthwiseHead(self.hm_head_channels, self.num_fg, 3,
                                           conv_relu=self.relu_before_depthwise,
                                           depth_kernel_sizes=self.depth_kernel_sizes,
                                           depth_group=self.depth_group,
                                           use_deform=self.depth_deform))
        else:
            hm_layers.append(nn.Conv2d(self.hm_head_channels, self.num_fg, 3, padding=1))

        wh_layers = nn.Sequential(*wh_layers)
        hm_layers = nn.Sequential(*hm_layers)
        return upsample_layers, shortcut_layers, wh_layers, hm_layers

    def _init_b1b2_layers(self, inplanes, planes, conv_num, use_bn):
        b1b2_layers = []
        bias = not use_bn
        if conv_num == 1:
            b1b2_layers.append(nn.Conv2d(inplanes, planes, 3,
                                         padding=1, bias=bias))
            if use_bn:
                b1b2_layers.append(nn.BatchNorm2d(planes))
                b1b2_layers.append(nn.ReLU(inplace=True))
                b1b2_layers.append(nn.Conv2d(planes, planes, 1))
        elif conv_num == 2:
            b1b2_layers.append(nn.Conv2d(inplanes, planes, 3,
                                         padding=1, bias=bias))
            if use_bn:
                b1b2_layers.append(nn.BatchNorm2d(planes))
            b1b2_layers.append(nn.ReLU(inplace=True))
            b1b2_layers.append(nn.Conv2d(planes, planes, 3,
                                         padding=1))
        elif conv_num == -1:
            b1b2_layers.append(
                ModulatedDeformConvPack(inplanes, planes, 3,
                                        padding=1, bias=bias))
            if use_bn:
                b1b2_layers.append(nn.BatchNorm2d(planes))
                b1b2_layers.append(nn.ReLU(inplace=True))
                b1b2_layers.append(nn.Conv2d(planes, planes, 1))
        if self.b1_last_relu:
            b1b2_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*b1b2_layers)

    def _init_layers(self):
        # brach-1, 1/32 => 1/8.
        self.upsample_layers_b1, self.shortcut_layers_b1, self.wh_b1, self.hm_b1 = \
            self._init_branch_layers(self.planes_b1, self.inplanes[-1], self.inplanes[::-1][1:3])

        # brach-2, 1/16 => 1/4.
        self.upsample_layers_b2, self.shortcut_layers_b2, self.wh_b2, self.hm_b2 = \
            self._init_branch_layers(self.planes_b2, self.inplanes[-2], self.inplanes[::-1][2:])

        if self.b1_s16_conv_num != 0:
            self.b1b2_s16_layers = self._init_b1b2_layers(
                self.planes_b1[0], self.inplanes[-2], self.b1_s16_conv_num, self.b1_s16_bn)
            if self.b1_concat:
                self.b1b2_s16_down = nn.Conv2d(2 * self.inplanes[-2], self.inplanes[-2], 3,
                                               padding=1)
        if self.b1_s8_conv_num != 0:
            self.b1b2_s8_layers = self._init_b1b2_layers(
                self.planes_b1[1], self.planes_b2[0], self.b1_s8_conv_num, self.b1_s8_bn)
            if self.b1_concat:
                self.b1b2_s8_down = nn.Conv2d(2 * self.planes_b2[0], self.planes_b2[0], 3,
                                              padding=1)

        if self.use_b2_extra_layers:
            bias = not self.extra_layer_bn
            extra_layers = [nn.Conv2d(self.inplanes[-2], self.inplanes[-2], 3,
                                      padding=1, bias=bias)]
            if self.extra_layer_bn:
                extra_layers.append(nn.BatchNorm2d(self.inplanes[-2]))
            extra_layers.append(nn.ReLU(inplace=True))

            if self.extra_layer_num == 2:
                extra_layers.append(nn.Conv2d(self.inplanes[-2], self.inplanes[-2], 3,
                                              padding=1, bias=bias))
                if self.extra_layer_bn:
                    extra_layers.append(nn.BatchNorm2d(self.inplanes[-2]))
                extra_layers.append(nn.ReLU(inplace=True))
            self.b2_extra_layers = nn.Sequential(*extra_layers)

    def init_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         normal_init(m, std=0.01)
        #     if isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        for upsample_layers in [self.upsample_layers_b1, self.upsample_layers_b2]:
            for m in upsample_layers.modules():
                if isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

        for shortcut_layers in [self.shortcut_layers_b1, self.shortcut_layers_b2]:
            for m in shortcut_layers.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)

        bias_cls = bias_init_with_prob(0.01)
        for hm in [self.hm_b1, self.hm_b2]:
            for m in hm.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            if self.depthwise_hm:
                if self.depth_init_kaiming:
                    normal_init(hm[-1].conv, std=0.01, bias=bias_cls)
                    if not self.depth_deform:
                        kaiming_init(hm[-1].depth_conv)
                else:
                    if self.depth_deform:
                        if hasattr(hm[-1].depth_conv, 'bias') and hm[
                            -1].depth_conv.bias is not None:
                            nn.init.constant_(hm[-1].depth_conv.bias, bias_cls)
                    else:
                        normal_init(hm[-1].depth_conv, std=0.01, bias=bias_cls)
            else:
                normal_init(hm[-1], std=0.01, bias=bias_cls)

        for wh in [self.wh_b1, self.wh_b2]:
            for m in wh.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

        if self.b1_s8_conv_num != 0:
            for m in self.b1b2_s8_layers.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
                if isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
            if self.b1_concat:
                for m in self.b1b2_s8_down.modules():
                    if isinstance(m, nn.Conv2d):
                        normal_init(m, std=0.01)
                    if isinstance(m, nn.BatchNorm2d):
                        constant_init(m, 1)
        if self.b1_s16_conv_num != 0:
            for m in self.b1b2_s16_layers.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
                if isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
            if self.b1_concat:
                for m in self.b1b2_s16_down.modules():
                    if isinstance(m, nn.Conv2d):
                        normal_init(m, std=0.01)
                    if isinstance(m, nn.BatchNorm2d):
                        constant_init(m, 1)

        if self.use_b2_extra_layers:
            for m in self.b2_extra_layers:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)

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
        x_b1 = []
        x = feats[-1]
        for i, upsampling_layer in enumerate(self.upsample_layers_b1):
            x = upsampling_layer(x)
            if i < len(self.shortcut_layers_b1):
                shortcut = self.shortcut_layers_b1[i](feats[-i - 2])
                x = x + shortcut
                x_b1.append(x)

        hm_b1 = self.hm_b1(x)
        wh_b1 = F.relu(self.wh_b1(x)) * self.wh_scale_factor_b1

        x = feats[-2]
        if self.use_b2_extra_layers:
            x = self.b2_extra_layers(x)
        if self.b1_s16_conv_num != 0:
            if self.b1_concat:
                y = torch.cat([x, self.b1b2_s16_layers(x_b1[0])], dim=1)
                x = self.b1b2_s16_down(y)
            elif self.b1_minus:
                x = x - self.b1b2_s16_layers(x_b1[0])
            else:
                x = x + self.b1b2_s16_layers(x_b1[0])
        for i, upsampling_layer in enumerate(self.upsample_layers_b2):
            x = upsampling_layer(x)
            if i < len(self.shortcut_layers_b2):
                shortcut = self.shortcut_layers_b2[i](feats[-i - 3])
                x = x + shortcut
                if self.b1_s8_conv_num != 0 and i == 0:
                    if self.b1_concat:
                        y = torch.cat([x, self.b1b2_s8_layers(x_b1[1])], dim=1)
                        x = self.b1b2_s8_down(y)
                    elif self.b1_minus:
                        x = x - self.b1b2_s8_layers(x_b1[1])
                    else:
                        x = x + self.b1b2_s8_layers(x_b1[1])

        hm_b2 = self.hm_b2(x)
        wh_b2 = F.relu(self.wh_b2(x)) * self.wh_scale_factor_b2

        return hm_b1, hm_b2, wh_b1, wh_b2

    def get_bboxes_single(self,
                          pred_hm,
                          pred_wh,
                          down_ratio,
                          topk):
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

        wh = wh.view(batch, topk, 4)
        clses = clses.view(batch, topk, 1).float()
        scores = scores.view(batch, topk, 1)

        bboxes = torch.cat([
            xs - wh[..., [0]], ys - wh[..., [1]], xs + wh[..., [2]],
            ys + wh[..., [3]]
        ], dim=2)
        return heat, inds, clses, scores, bboxes, xs, ys

    @force_fp32(apply_to=('pred_hm_b1', 'pred_hm_b2', 'pred_wh_b1', 'pred_wh_b2'))
    def get_bboxes(self,
                   pred_hm_b1,
                   pred_hm_b2,
                   pred_wh_b1,
                   pred_wh_b2,
                   img_metas,
                   cfg,
                   rescale=False):
        if self.get_bboxesv2:
            return self.get_bboxes_v2(
                pred_hm_b1, pred_hm_b2, pred_wh_b1, pred_wh_b2, img_metas, cfg, rescale=False)
        else:
            return self.get_bboxes_v1(
                pred_hm_b1, pred_hm_b2, pred_wh_b1, pred_wh_b2, img_metas, cfg, rescale=False)

    @force_fp32(apply_to=('pred_hm_b1', 'pred_hm_b2', 'pred_wh_b1', 'pred_wh_b2'))
    def get_bboxes_v2(self,
                      pred_hm_b1,
                      pred_hm_b2,
                      pred_wh_b1,
                      pred_wh_b2,
                      img_metas,
                      cfg,
                      rescale=False):
        topk = getattr(cfg, 'max_per_img', 100)
        heat_b1, inds_b1, clses_b1, scores_b1, bboxes_b1, xs_b1, ys_b1 = self.get_bboxes_single(
            pred_hm_b1, pred_wh_b1, self.down_ratio_b1, topk)
        heat_b2, inds_b2, clses_b2, scores_b2, bboxes_b2, xs_b2, ys_b2 = self.get_bboxes_single(
            pred_hm_b2, pred_wh_b2, self.down_ratio_b2, topk)

        result_list = []
        score_thr = getattr(cfg, 'score_thr', 0.01)
        for batch_i in range(heat_b1.shape[0]):
            scores_b1_per_img = scores_b1[batch_i]
            scores_b2_per_img = scores_b2[batch_i]
            scores_b1_keep = (scores_b1_per_img > score_thr).squeeze(-1)
            scores_b2_keep = (scores_b2_per_img > score_thr).squeeze(-1)

            scores_b1_per_img = scores_b1_per_img[scores_b1_keep]
            bboxes_b1_per_img = bboxes_b1[batch_i][scores_b1_keep]
            labels_b1_per_img = clses_b1[batch_i][scores_b1_keep].squeeze(-1).int()
            scores_b2_per_img = scores_b2_per_img[scores_b2_keep]
            bboxes_b2_per_img = bboxes_b2[batch_i][scores_b2_keep]
            labels_b2_per_img = clses_b2[batch_i][scores_b2_keep].squeeze(-1).int()

            img_shape = img_metas[batch_i]['pad_shape']
            bboxes_b1_per_img[:, 0::2] = bboxes_b1_per_img[:, 0::2].clamp(
                min=0, max=img_shape[1] - 1)
            bboxes_b1_per_img[:, 1::2] = bboxes_b1_per_img[:, 1::2].clamp(
                min=0, max=img_shape[0] - 1)
            bboxes_b2_per_img[:, 0::2] = bboxes_b2_per_img[:, 0::2].clamp(
                min=0, max=img_shape[1] - 1)
            bboxes_b2_per_img[:, 1::2] = bboxes_b2_per_img[:, 1::2].clamp(
                min=0, max=img_shape[0] - 1)

            unique_cls_ids_b1 = torch.unique(labels_b1_per_img)
            unique_cls_ids_b2 = torch.unique(labels_b2_per_img)
            print(unique_cls_ids_b1, unique_cls_ids_b2)
            # bboxes_per_img_per_cls = bboxes_per_img.new_zeros((0, 5))
            # labels_per_img_per_cls = labels_int_flatten.new_zeros((0,))
            # for cls_id in unique_cls_ids:
            #     cls_id_idx = (labels_int_flatten == cls_id)
            #     soft_bboxes, ori_idx = soft_nms(torch.cat((
            #         bboxes_per_img[cls_id_idx], scores_per_img[cls_id_idx]), dim=1),
            #         iou_thr=0.6)
            #     unique_labels = labels_int_flatten[cls_id_idx][ori_idx]
            #     bboxes_per_img_per_cls = torch.cat((bboxes_per_img_per_cls, soft_bboxes), dim=0)
            #     labels_per_img_per_cls = torch.cat((labels_per_img_per_cls, unique_labels))
            # bboxes_per_img = bboxes_per_img_per_cls
            # labels_per_img = labels_per_img_per_cls.float()
            #
            # if rescale:
            #     scale_factor = img_metas[batch_i]['scale_factor']
            #     bboxes_per_img /= bboxes_per_img.new_tensor(scale_factor)
            #
            # labels_per_img = labels_per_img.squeeze(-1)
            # result_list.append((bboxes_per_img, labels_per_img))

        return result_list

    @force_fp32(apply_to=('pred_hm_b1', 'pred_hm_b2', 'pred_wh_b1', 'pred_wh_b2'))
    def get_bboxes_v1(self,
                      pred_hm_b1,
                      pred_hm_b2,
                      pred_wh_b1,
                      pred_wh_b2,
                      img_metas,
                      cfg,
                      rescale=False):
        topk = getattr(cfg, 'max_per_img', 100)
        heat_b1, inds_b1, clses_b1, scores_b1, bboxes_b1, xs_b1, ys_b1 = self.get_bboxes_single(
            pred_hm_b1, pred_wh_b1, self.down_ratio_b1, topk)
        heat_b2, inds_b2, clses_b2, scores_b2, bboxes_b2, xs_b2, ys_b2 = self.get_bboxes_single(
            pred_hm_b2, pred_wh_b2, self.down_ratio_b2, topk)

        result_list = []
        score_thr = getattr(cfg, 'score_thr', 0.01)
        if 'b2' not in self.inf_branch:
            bboxes = bboxes_b1
            scores = scores_b1
            clses = clses_b1
        elif 'b1' not in self.inf_branch:
            bboxes = bboxes_b2
            scores = scores_b2
            clses = clses_b2
        else:
            bboxes = torch.cat([bboxes_b1, bboxes_b2], dim=1)
            scores = torch.cat([scores_b1, scores_b2], dim=1)
            clses = torch.cat([clses_b1, clses_b2], dim=1)
        for batch_i in range(bboxes.shape[0]):
            scores_per_img = scores[batch_i]
            scores_keep = (scores_per_img > score_thr).squeeze(-1)

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
            if self.norm_log:
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
