import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init, constant_init
import math
import numpy as np

from mmdet.ops import ModulatedDeformConvPack, soft_nms
from mmdet.core import multi_apply, force_fp32, multiclass_nms
from mmdet.core.utils.summary import add_summary, get_local_step
from mmdet.core.bbox import bbox_overlaps
from mmdet.models.losses import ct_focal_loss, giou_loss
from mmdet.models.utils import (
    build_conv_layer, build_norm_layer, bias_init_with_prob, ConvModule, TridentConv2d)
from ..registry import HEADS
from .anchor_head import AnchorHead


class UpsamplingLayers(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 no_upsample=False,
                 vallina=False,
                 offset_mean=False):
        if vallina:
            if isinstance(vallina, int):
                padding = int((vallina - 1) / 2)
                dila = padding
                mdcn = nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    padding=padding,
                    dilation=dila)
            else:
                mdcn = nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    padding=1)
        elif conv_cfg:
            mdcn = build_conv_layer(conv_cfg, in_channels, out_channels)
        else:
            mdcn = ModulatedDeformConvPack(
                in_channels,
                out_channels,
                3,
                offset_mean=offset_mean,
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

    def __init__(self, in_channels, out_channels, kernel_sizes, conv_cfg,
                 down=False, down_sec=False):
        super(ShortcutConnection, self).__init__()
        layers = []
        for i, kernel_size in enumerate(kernel_sizes):
            inc = in_channels if i == 0 else out_channels
            stride = 1
            if (down and i == 0) or (down_sec and i == 1):
                stride = 2
            padding = (kernel_size - 1) // 2
            if conv_cfg:
                layers.append(
                    build_conv_layer(conv_cfg, inc, out_channels, kernel_size,
                                     stride=stride, padding=padding))
            else:
                layers.append(
                    nn.Conv2d(inc, out_channels, kernel_size,
                              stride=stride, padding=padding))
            if i < len(kernel_sizes) - 1:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ASFF(nn.Module):
    def __init__(self, level):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [256, 128, 64]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = ConvModule(
                128,
                self.inter_dim,
                3,
                padding=1,
                stride=2,
                norm_cfg=dict(type='BN'))
            self.stride_level_2 = ConvModule(
                64,
                self.inter_dim,
                3,
                padding=1,
                stride=2,
                norm_cfg=dict(type='BN'))
            self.expand = ConvModule(
                self.inter_dim,
                256,
                3,
                padding=1,
                norm_cfg=dict(type='BN'))
        elif level == 1:
            self.compress_level_0 = ConvModule(
                256,
                self.inter_dim,
                1,
                norm_cfg=dict(type='BN'))
            self.stride_level_2 = ConvModule(
                64,
                self.inter_dim,
                3,
                padding=1,
                stride=2,
                norm_cfg=dict(type='BN'))
            self.expand = ConvModule(
                self.inter_dim,
                128,
                3,
                padding=1,
                norm_cfg=dict(type='BN'))
        elif level == 2:
            self.compress_level_0 = ConvModule(
                256,
                self.inter_dim,
                1,
                norm_cfg=dict(type='BN'))
            self.compress_level_1 = ConvModule(
                128,
                self.inter_dim,
                1,
                norm_cfg=dict(type='BN'))
            self.expand = ConvModule(
                self.inter_dim,
                64,
                3,
                padding=1,
                norm_cfg=dict(type='BN'))

        compress_c = 16

        self.weight_level_0 = ConvModule(self.inter_dim, compress_c, 1, norm_cfg=dict(type='BN'))
        self.weight_level_1 = ConvModule(self.inter_dim, compress_c, 1, norm_cfg=dict(type='BN'))
        self.weight_level_2 = ConvModule(self.inter_dim, compress_c, 1, norm_cfg=dict(type='BN'))

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        print(levels_weight)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)
        return out


@HEADS.register_module
class TTFv3Head(AnchorHead):

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 shortcut_cfg=(1, 2, 3),
                 use_extra_shortcut=True,
                 s16_shortcut_twice=False,
                 shortcut_conv_cfg=None,
                 up_conv_cfg=None,
                 upsample_vallina=False,
                 dcn_offset_mean=False,
                 down_ratio=(8, 4),
                 fake_s8=False,
                 with_fpn=False,
                 length_range=((64, 512), (1, 64)),
                 anchor_ratios=None,
                 hm_head_channels=((128, 128), (64, 64)),
                 wh_head_channels=((32, 32), (32, 32)),
                 num_classes=81,
                 wh_scale_factor=(8., 8.),
                 use_asff=False,
                 alpha=0.6,
                 beta=0.6,
                 hm_weight=(1.4, 1.),
                 wh_weight=(7., 5.),
                 train_branch=(True, True),
                 inf_branch=(True, True),
                 train_warmup_all=False,
                 use_simple_nms=True,
                 fast_nms=False,
                 use_fast_nms_v2=False,
                 trand_nms=False,
                 max_objs=128,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(AnchorHead, self).__init__()
        assert len(inplanes) == 4 and len(planes) == 3 and len(shortcut_cfg) == 3
        assert len(down_ratio) == len(hm_head_channels) == len(wh_head_channels) == \
               len(wh_scale_factor) == \
               len(hm_weight) == len(wh_weight) == len(length_range) == len(train_branch) == \
               len(inf_branch)
        assert len(down_ratio) in [1, 2, 3]
        self.inplanes = inplanes
        self.planes = planes
        self.shortcut_cfg = shortcut_cfg
        self.use_extra_shortcut = use_extra_shortcut
        self.s16_shortcut_twice = s16_shortcut_twice
        self.shortcut_conv_cfg = shortcut_conv_cfg
        self.up_conv_cfg = up_conv_cfg
        self.upsample_vallina = upsample_vallina
        self.dcn_offset_mean = dcn_offset_mean
        self.down_ratio = down_ratio
        self.fake_s8 = fake_s8
        self.with_fpn = with_fpn

        self.auto_range = False if isinstance(length_range[0], (list, tuple)) else True
        self.length_range = length_range
        if anchor_ratios:
            assert self.auto_range
        elif self.auto_range:
            anchor_ratios = (1.,)
        self.anchor_ratios = anchor_ratios
        self.hm_head_channels = hm_head_channels
        self.wh_head_channels = wh_head_channels
        self.num_classes = num_classes
        self.num_fg = num_classes - 1
        self.wh_scale_factor = wh_scale_factor
        if use_asff:
            assert len(down_ratio) == 3
        self.use_asff = use_asff
        self.alpha = alpha
        self.beta = beta
        self.max_objs = max_objs
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.train_branch = train_branch
        self.inf_branch = inf_branch
        self.train_warmup_all = train_warmup_all
        self.use_simple_nms = use_simple_nms
        self.fast_nms = fast_nms
        self.use_fast_nms_v2 = use_fast_nms_v2
        self.trand_nms = trand_nms
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self._init_layers()

    def get_down_ratio(self, down_ratio):
        if down_ratio == 8 and self.fake_s8:
            return 4
        return down_ratio

    def _init_branch_layers(self, planes, idx=-1):
        wh_layers, hm_layers = [], []
        inp = planes
        for outp in self.wh_head_channels[idx]:
            wh_layers.append(
                ConvModule(
                    inp,
                    outp,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg))
            inp = outp
        wh_layers.append(nn.Conv2d(inp, 4, 3, padding=1))

        inp = planes
        for outp in self.hm_head_channels[idx]:
            hm_layers.append(
                ConvModule(
                    inp,
                    outp,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg))
            inp = outp
        hm_layers.append(nn.Conv2d(inp, self.num_fg, 3, padding=1))

        wh_layers = nn.Sequential(*wh_layers)
        hm_layers = nn.Sequential(*hm_layers)
        return wh_layers, hm_layers

    def _init_extra_layers(self):
        self.s8_upsample_layers = nn.ModuleList()
        self.s8_shortcut_layers = nn.ModuleList()
        self.s16_upsample_layers = nn.ModuleList()
        self.s16_shortcut_layers = nn.ModuleList()

        if 8 in self.down_ratio:
            self.s8_upsample_layers.append(
                UpsamplingLayers(self.planes[1], self.planes[1],
                                 conv_cfg=self.up_conv_cfg,
                                 norm_cfg=self.norm_cfg, no_upsample=not self.fake_s8,
                                 vallina=self.upsample_vallina, offset_mean=self.dcn_offset_mean))
            self.s8_shortcut_layers.append(
                ShortcutConnection(
                    self.inplanes[0], self.planes[1], [3] * self.shortcut_cfg[2],
                    self.shortcut_conv_cfg, down=not self.fake_s8))
        if 16 in self.down_ratio:
            self.s16_upsample_layers.extend([
                UpsamplingLayers(self.planes[0], self.planes[0],
                                 conv_cfg=self.up_conv_cfg,
                                 norm_cfg=self.norm_cfg, no_upsample=True,
                                 vallina=self.upsample_vallina, offset_mean=self.dcn_offset_mean),
                UpsamplingLayers(self.planes[0], self.planes[0],
                                 conv_cfg=self.up_conv_cfg,
                                 norm_cfg=self.norm_cfg, no_upsample=True,
                                 vallina=self.upsample_vallina, offset_mean=self.dcn_offset_mean),
            ])
            self.s16_shortcut_layers.append(
                ShortcutConnection(
                    self.inplanes[1], self.planes[0], [3] * self.shortcut_cfg[1],
                    self.shortcut_conv_cfg, down=True))
            if self.s16_shortcut_twice:
                self.s16_shortcut_layers.append(
                    ShortcutConnection(
                        self.inplanes[0], self.planes[0], [3] * self.shortcut_cfg[2],
                        self.shortcut_conv_cfg, down=True, down_sec=True))

    def _init_extra_heads(self, idx):
        self.s8_wh = nn.ModuleList()
        self.s8_hm = nn.ModuleList()
        self.s16_wh = nn.ModuleList()
        self.s16_hm = nn.ModuleList()

        if 8 in self.down_ratio:
            self.s8_wh, self.s8_hm = self._init_branch_layers(self.planes[-2], idx=idx)
            idx -= 1

        if 16 in self.down_ratio:
            self.s16_wh, self.s16_hm = self._init_branch_layers(self.planes[-3], idx=idx)

    def _init_layers(self):
        self.upsample_layers = nn.ModuleList([
            UpsamplingLayers(
                self.inplanes[-1], self.planes[0], conv_cfg=self.up_conv_cfg,
                norm_cfg=self.norm_cfg, vallina=self.upsample_vallina,
                offset_mean=self.dcn_offset_mean),
            UpsamplingLayers(
                self.planes[0], self.planes[1], conv_cfg=self.up_conv_cfg,
                norm_cfg=self.norm_cfg, vallina=self.upsample_vallina,
                offset_mean=self.dcn_offset_mean),
            UpsamplingLayers(
                self.planes[1], self.planes[2], conv_cfg=self.up_conv_cfg,
                norm_cfg=self.norm_cfg, vallina=self.upsample_vallina,
                offset_mean=self.dcn_offset_mean)
        ])

        self.shortcut_layers = nn.ModuleList([
            ShortcutConnection(self.inplanes[2], self.planes[0], [3] * self.shortcut_cfg[0],
                               self.shortcut_conv_cfg),
            ShortcutConnection(self.inplanes[1], self.planes[1], [3] * self.shortcut_cfg[1],
                               self.shortcut_conv_cfg),
            ShortcutConnection(self.inplanes[0], self.planes[2], [3] * self.shortcut_cfg[2],
                               self.shortcut_conv_cfg)
        ])
        self._init_extra_layers()

        # brach-1/4.
        idx = -1
        self.s4_wh, self.s4_hm = self._init_branch_layers(self.planes[-1], idx=idx)
        if 4 in self.down_ratio:
            idx -= 1
        self._init_extra_heads(idx)

        if self.use_asff:
            self.s16_asff = ASFF(0)
            self.s8_asff = ASFF(1)
            self.s4_asff = ASFF(2)

    def init_weights(self):
        for upsample_layers in [self.upsample_layers, self.s8_upsample_layers,
                                self.s16_upsample_layers]:
            for m in upsample_layers.modules():
                if isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

        for shortcut_layers in [self.shortcut_layers, self.s8_shortcut_layers,
                                self.s16_shortcut_layers]:
            for m in shortcut_layers.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)

        bias_cls = bias_init_with_prob(0.01)
        for hm in [self.s4_hm, self.s8_hm, self.s16_hm]:
            for m in hm.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            if len(hm) > 0:
                normal_init(hm[-1], std=0.01, bias=bias_cls)

        for wh in [self.s4_wh, self.s8_wh, self.s16_wh]:
            for m in wh.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

        for m in self.modules():
            if isinstance(m, ModulatedDeformConvPack):
                if hasattr(m, 'conv_offset_mask'):
                    constant_init(m.conv_offset_mask, 0)
                else:
                    constant_init(m.conv_offset, 0)

        if self.use_asff:
            for asff in [self.s16_asff, self.s8_asff, self.s4_asff]:
                for m in asff.modules():
                    if isinstance(m, nn.Conv2d):
                        normal_init(m, std=0.01)
                    elif isinstance(m, nn.BatchNorm2d):
                        constant_init(m, 1)

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: list(tensor), (batch, 80, h, w).
            wh: list(tensor), (batch, 4, h, w).
        """
        if self.with_fpn:
            feats = feats[::-1]
        else:
            y, shortcuts = [], []
            x = feats[-1]
            for i, shortcut_layer in enumerate(self.shortcut_layers):
                shortcuts.append(shortcut_layer(feats[-i - 2]))

            for i, upsampling_layer in enumerate(self.upsample_layers):
                x = upsampling_layer(x)
                x = x + shortcuts[i]
                y.append(x)

        idx = 0
        if 16 in self.down_ratio:
            if self.with_fpn:
                y_s16 = feats[idx]
            else:
                y_s16 = y[0]
                y_s16 = self.s16_upsample_layers[0](y_s16)
                if self.use_extra_shortcut:
                    y_s16 = y_s16 + self.s16_shortcut_layers[0](feats[1])
                y_s16 = self.s16_upsample_layers[1](y_s16)
                if self.s16_shortcut_twice:
                    y_s16 = y_s16 + self.s16_shortcut_layers[1](feats[0])
            idx += 1

        if 8 in self.down_ratio:
            if self.with_fpn:
                y_s8 = feats[idx]
            else:
                y_s8 = y[1]
                y_s8 = self.s8_upsample_layers[0](y_s8)
                if self.use_extra_shortcut:
                    y_s8 = y_s8 + self.s8_shortcut_layers[0](feats[0])
            idx += 1

        if 4 in self.down_ratio:
            if self.with_fpn:
                y_s4 = feats[idx]
            else:
                y_s4 = y[-1]

        if self.use_asff:
            y_s16 = self.s16_asff(y_s16, y_s8, y_s4)
            y_s8 = self.s8_asff(y_s16, y_s8, y_s4)
            y_s4 = self.s4_asff(y_s16, y_s8, y_s4)

        hm, wh = [], []
        idx = 0
        if 16 in self.down_ratio:
            hm.append(self.s16_hm(y_s16))
            wh.append(F.relu(self.s16_wh(y_s16)) * self.wh_scale_factor[idx])
            idx += 1

        if 8 in self.down_ratio:
            hm.append(self.s8_hm(y_s8))
            wh.append(F.relu(self.s8_wh(y_s8)) * self.wh_scale_factor[idx])
            idx += 1

        if 4 in self.down_ratio:
            hm.append(self.s4_hm(y_s4))
            wh.append(F.relu(self.s4_wh(y_s4)) * self.wh_scale_factor[idx])

        return hm, wh

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

    def add_to_list(self, inp, outp):
        for i, o in zip(inp, outp):
            o.append(i)
        return outp

    @force_fp32(apply_to=('pred_hm', 'pred_wh'))
    def get_bboxes(self,
                   pred_hm,
                   pred_wh,
                   img_metas,
                   cfg,
                   rescale=False):
        topk = getattr(cfg, 'max_per_img', 100)

        results = [[], [], [], [], [], [], []]

        idx = 0
        if 16 in self.down_ratio and self.inf_branch[idx]:
            down_ratio = self.get_down_ratio(self.down_ratio[idx])
            s16_out = self.get_bboxes_single(pred_hm[idx], pred_wh[idx], down_ratio, topk)
            results = self.add_to_list(s16_out, results)
            idx += 1
        if 8 in self.down_ratio and self.inf_branch[idx]:
            down_ratio = self.get_down_ratio(self.down_ratio[idx])
            s8_out = self.get_bboxes_single(pred_hm[idx], pred_wh[idx], down_ratio, topk)
            results = self.add_to_list(s8_out, results)
            idx += 1
        if 4 in self.down_ratio and self.inf_branch[idx]:
            down_ratio = self.get_down_ratio(self.down_ratio[idx])
            s4_out = self.get_bboxes_single(pred_hm[idx], pred_wh[idx], down_ratio, topk)
            results = self.add_to_list(s4_out, results)

        heat, inds, clses, scores, bboxes, xs, ys = results

        result_list = []
        score_thr = getattr(cfg, 'score_thr', 0.01)
        for batch_i in range(heat[0].shape[0]):
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

                b1_retains, b2_retains = [], []
                b1_keeps, b2_keeps = [], []
                b1_erases, b2_erases = [], []
                for idx in range(len(scores_per_img) - 1):
                    heat_b1, heat_b2 = heat[idx], heat[idx + 1]
                    bboxes_b1_per_img, bboxes_b2_per_img = bboxes_per_img[idx], \
                                                           bboxes_per_img[idx + 1]
                    labels_b1_per_img, labels_b2_per_img = labels_per_img[idx], \
                                                           labels_per_img[idx + 1]
                    scores_b1_per_img, scores_b2_per_img = scores_per_img[idx], \
                                                           scores_per_img[idx + 1]
                    duplicate_cls_b1 = heat_b1.new_ones(
                        (bboxes_b1_per_img.shape[0], bboxes_b2_per_img.shape[0]),
                        dtype=torch.long) * labels_b1_per_img.unsqueeze(-1)
                    duplicate_cls_b2 = heat_b1.new_ones(
                        (bboxes_b1_per_img.shape[0], bboxes_b2_per_img.shape[0]),
                        dtype=torch.long) * labels_b2_per_img.unsqueeze(0)
                    duplicate_cls = (duplicate_cls_b1 == duplicate_cls_b2)

                    b1_keep = bboxes_b1_per_img.new_ones((bboxes_b1_per_img.shape[0],),
                                                         dtype=torch.bool)
                    b2_keep = bboxes_b2_per_img.new_ones((bboxes_b2_per_img.shape[0],),
                                                         dtype=torch.bool)
                    if self.use_fast_nms_v2:
                        b1_retain = bboxes_b1_per_img.new_ones((bboxes_b1_per_img.shape[0],),
                                                               dtype=torch.bool)
                        b2_retain = bboxes_b2_per_img.new_ones((bboxes_b2_per_img.shape[0],),
                                                               dtype=torch.bool)
                        if duplicate_cls.any():
                            ious_large = bbox_overlaps(bboxes_b1_per_img,
                                                       bboxes_b2_per_img) > 0.6
                            N = ious_large & duplicate_cls
                            H_b1, H_b2 = N.any(1), N.any(0)
                            N = N.float()
                            M_b1 = (scores_b2_per_img.view(1, -1) * N).max(1)[0]
                            M_b2 = (scores_b1_per_img * N).max(0)[0]
                            b1_large = (scores_b1_per_img.view(-1) > M_b1)
                            b2_large = (scores_b2_per_img.view(-1) > M_b2)

                            b1_retain, b2_retain = (b1_large & H_b1), (b2_large & H_b2)
                            b1_keep, b2_keep = (b1_large & ~H_b1), (b2_large & ~H_b2)
                        b1_retains.append(b1_retain)
                        b2_retains.append(b2_retain)
                    else:
                        if duplicate_cls.any():
                            ious_large = bbox_overlaps(bboxes_b1_per_img,
                                                       bboxes_b2_per_img) > 0.6
                            duplicate = ious_large & duplicate_cls
                            scores_b2_max, b2_max_loc = (scores_b2_per_img.view(
                                1, -1) * duplicate.float()).max(1)
                            b1_keep = (scores_b1_per_img.view(-1) >= scores_b2_max)
                            b2_max_loc_keep = b2_max_loc[~b1_keep]
                            b2_max_loc_keep_hot = heat_b1.new_zeros((duplicate.shape[1],),
                                                                    dtype=torch.bool)
                            b2_max_loc_keep_hot.scatter(0, b2_max_loc_keep, 1)
                            b2_keep = ~(b1_keep.view(-1, 1) * duplicate).any(0)
                            b2_keep = b2_keep | b2_max_loc_keep_hot

                    b1_keeps.append(b1_keep)
                    b2_keeps.append(b2_keep)

                if self.use_fast_nms_v2:
                    level_retain = [b1_retains[0] | b1_keeps[0]]
                    for idx in range(len(self.down_ratio) - 2):
                        retain = b1_retains[idx + 1] | b2_retains[idx]
                        retain = retain | (b1_keeps[idx + 1] & b2_keeps[idx])
                        level_retain.append(retain)
                    level_retain.append(b2_retains[-1] | b2_keeps[-1])
                else:
                    level_retain = [b1_keeps[0]]
                    for idx in range(len(self.down_ratio) - 2):
                        level_retain.append(b1_keeps[idx + 1] & b2_keeps[idx])
                    level_retain.append(b2_keeps[-1])

                scores_per_img = torch.cat([score_per_img[keep] for (score_per_img, keep)
                                            in zip(scores_per_img, level_retain)], dim=0)
                bboxes_per_img = torch.cat([bbox_per_img[keep] for (bbox_per_img, keep)
                                            in zip(bboxes_per_img, level_retain)], dim=0)
                labels_per_img = torch.cat([label_per_img[keep] for (label_per_img, keep)
                                            in zip(labels_per_img, level_retain)], dim=0)
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

    def loss_single(self,
                    pred_hm,
                    pred_wh,
                    heatmap,
                    box_target,
                    wh_weight,
                    down_ratio,
                    hm_weight_factor,
                    wh_weight_factor):
        H, W = pred_hm.shape[2:]
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        loss_cls = ct_focal_loss(pred_hm, heatmap) * hm_weight_factor

        base_step = self.get_down_ratio(down_ratio)
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
        base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        # (batch, h, w, 4)
        pred_boxes = torch.cat((base_loc - pred_wh[:, [0, 1]],
                                base_loc + pred_wh[:, [2, 3]]),
                               dim=1).permute(0, 2, 3, 1)
        boxes = box_target.permute(0, 2, 3, 1)

        mask = wh_weight.view(-1, H, W)
        avg_factor = mask.sum() + 1e-4

        loss_bbox = giou_loss(
            pred_boxes, boxes, mask, avg_factor=avg_factor) * wh_weight_factor

        return loss_cls, loss_bbox

    @force_fp32(apply_to=('pred_hm', 'pred_wh'))
    def loss(self,
             pred_hm,
             pred_wh,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        heatmap, box_target, reg_weight = self.ttf_target(
            gt_bboxes, gt_labels, img_metas)

        loss_cls, loss_bbox = multi_apply(
            self.loss_single,
            pred_hm,
            pred_wh,
            heatmap,
            box_target,
            reg_weight,
            self.down_ratio,
            self.hm_weight,
            self.wh_weight)

        warmup_stage = self.train_warmup_all and get_local_step() <= 500
        loss_dict = dict()

        for idx, down_ratio in enumerate(self.down_ratio):
            if not self.train_branch[idx] and not warmup_stage:
                loss_cls[idx] = loss_cls[idx].fill_(0)
                loss_bbox[idx] = loss_bbox[idx].fill_(0)
            loss_dict['losses/ttfnetv2_loss_hm_s{}'.format(down_ratio)] = loss_cls[idx]
            loss_dict['losses/ttfnetv2_loss_wh_s{}'.format(down_ratio)] = loss_bbox[idx]

        return loss_dict

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
                                 gt_level_idx,
                                 output_h,
                                 output_w,
                                 down_ratio,
                                 gt_boxes,
                                 gt_labels,
                                 boxes_ind,
                                 boxes_area_topk_log):
        boxes_area_topk_log = boxes_area_topk_log[gt_level_idx]
        gt_boxes = gt_boxes[gt_level_idx]
        gt_labels = gt_labels[gt_level_idx]
        boxes_ind = boxes_ind[gt_level_idx]

        down_ratio = self.get_down_ratio(down_ratio)
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

    def ttf_target_single(self, gt_boxes, gt_labels, pad_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            pad_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w).
            reg_weight: tensor, same as box_target
        """
        heatmap_channel = self.num_fg

        boxes_areas_log = self.bbox_areas(gt_boxes).log()
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log,
                                                    boxes_areas_log.size(0))

        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        if self.auto_range:
            anchors_range = gt_boxes.new_tensor(self.length_range).view(1, -1, 1)
            anchors_range = anchors_range.repeat(len(self.anchor_ratios), 1, 4)
            anchors_range[:, :, [0, 1]] = -anchors_range[:, :, [0, 1]] / 2
            anchors_range[:, :, [2, 3]] = anchors_range[:, :, [2, 3]] / 2
            anchor_ratio_scale = torch.sqrt(gt_boxes.new_tensor(self.anchor_ratios)).view(-1, 1, 1)
            anchors_range[:, :, [0, 2]] /= anchor_ratio_scale
            anchors_range[:, :, [1, 3]] *= anchor_ratio_scale
            anchors_range = anchors_range.view(-1, 4)

            ct_align_boxes = gt_boxes.new_tensor(gt_boxes)
            ct_align_boxes[:, [0, 2]] -= (ct_align_boxes[:, [0]] + ct_align_boxes[:, [2]]) / 2
            ct_align_boxes[:, [1, 3]] -= (ct_align_boxes[:, [1]] + ct_align_boxes[:, [3]]) / 2
            overlaps = bbox_overlaps(ct_align_boxes, anchors_range)
            _, ind = overlaps.max(1)
            ind = ind % len(self.length_range)

        heatmap, fake_heatmap, box_target, reg_weight = [], [], [], []
        output_hs, output_ws, gt_level_idx = [], [], []
        for i, (down_ratio) in enumerate(self.down_ratio):
            down_ratio = self.get_down_ratio(down_ratio)
            output_h, output_w = [shape // down_ratio for shape in pad_shape]
            heatmap.append(gt_boxes.new_zeros((heatmap_channel, output_h, output_w)))
            fake_heatmap.append(gt_boxes.new_zeros((output_h, output_w)))
            box_target.append(gt_boxes.new_ones((4, output_h, output_w)) * -1)
            reg_weight.append(gt_boxes.new_zeros((1, output_h, output_w)))

            output_hs.append(output_h)
            output_ws.append(output_w)
            if not self.auto_range:
                gt_level_idx.append(
                    (boxes_area_topk_log >= math.log(self.length_range[i][0] ** 2)) &
                    (boxes_area_topk_log <= math.log(self.length_range[i][1] ** 2)))
            else:
                gt_level_idx.append(ind == i)

        if len(gt_level_idx) == 2:
            add_summary('gt_num', b1=gt_level_idx[0].sum().cpu().item(),
                        b2=gt_level_idx[1].sum().cpu().item())
        elif len(gt_level_idx) == 3:
            add_summary('gt_num', b1=gt_level_idx[0].sum().cpu().item(),
                        b2=gt_level_idx[1].sum().cpu().item(),
                        b3=gt_level_idx[2].sum().cpu().item())

        heatmap, box_target, reg_weight = multi_apply(
            self.ttf_target_single_single,
            heatmap,
            box_target,
            reg_weight,
            fake_heatmap,
            gt_level_idx,
            output_hs,
            output_ws,
            self.down_ratio,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            boxes_ind=boxes_ind,
            boxes_area_topk_log=boxes_area_topk_log)

        return heatmap, box_target, reg_weight

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
            pad_shape = (img_metas[0]['pad_shape'][0], img_metas[0]['pad_shape'][1])

            heatmap, box_target, reg_weight = multi_apply(
                self.ttf_target_single,
                gt_boxes,
                gt_labels,
                pad_shape=pad_shape)

            s_heatmap, s_box_target, s_reg_weight = [], [], []
            for level_i in range(len(heatmap[0])):
                s_heatmap.append(torch.stack([h[level_i] for h in heatmap], dim=0).detach())
                s_box_target.append(torch.stack([t[level_i] for t in box_target], dim=0).detach())
                s_reg_weight.append(torch.stack([t[level_i] for t in reg_weight], dim=0).detach())

            return s_heatmap, s_box_target, s_reg_weight

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
