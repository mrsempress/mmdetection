import numpy as np
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init

from mmdet.ops import ModulatedDeformConvPack, ModulatedDeformConv, DeformConvPack
from mmdet.core import multi_apply, bbox_areas, force_fp32, nms_agnostic
from mmdet.core.utils.common import gather_feat, tranpose_and_gather_feat
from mmdet.core.utils.summary import (
    add_summary, every_n_local_step, add_feature_summary, add_histogram_summary)
from mmdet.core.anchor.guided_anchor_target import calc_region
from mmdet.models.losses import ct_focal_loss, giou_loss, py_sigmoid_focal_loss, weighted_l1
from mmdet.models.utils import (
    build_norm_layer, gaussian_radius, draw_umich_gaussian, bias_init_with_prob,
    ConvModule, simple_nms, ShortcutConv2d, draw_truncate_gaussian, build_conv_layer,
    Scale)
from .anchor_head import AnchorHead
from ..registry import HEADS


@HEADS.register_module
class CXTHead(AnchorHead):

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 base_down_ratio=32,
                 head_conv=256,
                 wh_conv=32,
                 new_wh_generate=False,
                 hm_head_conv_num=1,
                 wh_head_conv_num=1,
                 ct_head_conv_num=1,
                 fill_small=False,
                 predict_together=False,
                 use_trident=False,
                 last_mdcn_to_conv=False,
                 use_deconv=False,
                 use_deconv_init=False,
                 norm_after_upsample=True,
                 fovea_hm=False,
                 num_classes=81,
                 use_exp_wh=False,
                 shortcut_kernel=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 use_shortcut=True,
                 shortcut_cfg=(1, 1, 1),
                 use_prelu=False,
                 activation_last=False,
                 shortcut_attention=(False, False, False),
                 hm_init_value=-6.0,
                 wh_001=False,
                 norm_wh=False,
                 wh_offset_base=None,
                 only_merge=False,
                 avg_wh_weightv2=False,
                 avg_wh_weightv3=False,
                 avg_wh_weightv4=False,
                 ct_gaussian=False,
                 use_nms=False,
                 wh_area_process='log',
                 wh_agnostic=False,
                 wh_heatmap=False,
                 hm_center_ratio=None,
                 center_ratio=0.2,
                 ct_version=True,
                 giou_weight=1.,
                 merge_weight=0.,
                 hm_weight=1.,
                 ct_weight=0.3):
        super(AnchorHead, self).__init__()
        assert len(planes) in [2, 3, 4] and \
               len(planes) == len(shortcut_cfg) == len(shortcut_attention)
        assert wh_area_process in [None, 'norm', 'log', 'sqrt']
        self.down_ratio = base_down_ratio // 2 ** len(planes)
        self.planes = planes
        self.head_conv = head_conv
        self.num_classes = num_classes
        self.num_fg = num_classes - 1
        self.use_deconv = use_deconv
        self.fovea_hm = fovea_hm
        self.norm_after_upsample = norm_after_upsample
        self.predict_together = predict_together
        self.wh_001 = wh_001
        self.use_shortcut = use_shortcut
        self.use_nms = use_nms

        wh_planes = 4 * self.num_fg
        if wh_agnostic:
            wh_planes = 4
        self.wh_agnostic = wh_agnostic

        self.new_wh_generate = new_wh_generate
        self.use_trident = use_trident
        self.last_mdcn_to_conv = last_mdcn_to_conv
        self.use_exp_wh = use_exp_wh
        self.only_merge = only_merge
        self.conv_cfg = conv_cfg
        self.norm_wh = norm_wh

        self.wh_offset_base = wh_offset_base if wh_offset_base is not None else None
        if self.wh_offset_base == 'auto':
            self.wh_offset_base = Scale(4.0, use_pow=True)
        self.use_deconv_init = use_deconv_init

        self.hm_init_value = hm_init_value
        self.hm_weight = hm_weight
        self.fp16_enabled = False

        # repeat deconv n times. 32x to 4x by default.
        self.deconv_layers = nn.ModuleList([
            self._make_deconv_layer(inplanes[-1], 1, [planes[0]], [4], norm_cfg=norm_cfg),
            self._make_deconv_layer(planes[0], 1, [planes[1]], [4], norm_cfg=norm_cfg)
        ])
        for i in range(2, len(planes)):
            self.deconv_layers.append(
                self._make_deconv_layer(planes[i - 1], 1, [planes[i]], [4], norm_cfg=norm_cfg))

        padding = (shortcut_kernel - 1) // 2
        self.shortcut_layers = self._make_shortcut(
            inplanes[:-1][::-1][:len(planes)], planes, shortcut_cfg, shortcut_attention,
            use_prelu, activation_last, kernel_size=shortcut_kernel, padding=padding)

        # heads
        if not predict_together:
            self.wh = self._make_conv_layer(wh_planes, wh_head_conv_num, wh_conv)
            self.hm = self._make_conv_layer(self.num_fg, hm_head_conv_num)
        else:
            self.hmwh = self._make_conv_layer(wh_planes + self.num_fg,
                                              max(hm_head_conv_num, wh_head_conv_num))
        if fovea_hm and not only_merge:
            self.centerness = self._make_conv_layer(1, ct_head_conv_num)

        self._target_generator = CXTTargetGenerator(self.num_fg, wh_planes, fovea_hm,
                                                    avg_wh_weightv2, avg_wh_weightv3,
                                                    avg_wh_weightv4, only_merge,
                                                    self.wh_offset_base, fill_small,
                                                    wh_area_process, wh_heatmap, hm_center_ratio,
                                                    ct_gaussian,
                                                    down_ratio=self.down_ratio,
                                                    center_ratio=center_ratio,
                                                    wh_agnostic=wh_agnostic,
                                                    ct_version=ct_version)
        self._loss = CXTLoss(giou_weight, hm_weight, ct_weight, merge_weight, fovea_hm,
                             only_merge, ct_version, wh_agnostic=wh_agnostic,
                             down_ratio=self.down_ratio)

    def _make_shortcut(self,
                       inplanes,
                       planes,
                       shortcut_cfg,
                       shortcut_attention,
                       use_prelu,
                       activation_last,
                       kernel_size=3,
                       padding=1):
        assert len(inplanes) == len(planes) == len(shortcut_cfg) == len(shortcut_attention)

        shortcut_layers = nn.ModuleList()
        for (inp, outp, layer_num, attention) in zip(
                inplanes, planes, shortcut_cfg, shortcut_attention):
            assert layer_num > 0
            layer = ShortcutConv2d(
                inp, outp, [kernel_size] * layer_num, [padding] * layer_num, use_cbam=attention,
                activation_last=activation_last, use_prelu=use_prelu)
            shortcut_layers.append(layer)
        return shortcut_layers

    def _make_deconv_layer(self, inplanes, num_layers, num_filters, num_kernels, norm_cfg=None):
        """

        Args:
            inplanes: in-channel num.
            num_layers: deconv layer num.
            num_filters: out channel of the deconv layers.
            num_kernels: int
            norm_cfg: dict()

        Returns:
            stacked deconv layers.
        """
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            planes = num_filters[i]
            inplanes = inplanes if i == 0 else num_filters[i - 1]

            if self.use_trident:
                mdcn = build_conv_layer(dict(type='TriConv'), inplanes, planes)
            else:
                if self.last_mdcn_to_conv and i == num_layers - 1:
                    mdcn = nn.Conv2d(inplanes, planes, 3, padding=1, bias=False)
                else:
                    mdcn = ModulatedDeformConvPack(inplanes, planes, 3, stride=1,
                                                   padding=1, dilation=1, deformable_groups=1)
            if self.use_deconv:
                kernel, padding, output_padding = (4, 1, 0)
                up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False)
                self.fill_up_weights(up)
            else:
                up = nn.UpsamplingBilinear2d(scale_factor=2)

            layers.append(mdcn)
            if norm_cfg:
                layers.append(build_norm_layer(norm_cfg, planes)[1])
            layers.append(nn.ReLU(inplace=True))

            layers.append(up)
            if self.use_deconv or self.norm_after_upsample:
                if norm_cfg:
                    layers.append(build_norm_layer(norm_cfg, planes)[1])
                layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_conv_layer(self, out_channel, conv_num=1, head_conv_plane=None):
        head_convs = []
        if not self.new_wh_generate:
            head_conv_plane = self.head_conv if not head_conv_plane else head_conv_plane
            for i in range(conv_num):
                inp = self.planes[-1] if i == 0 else head_conv_plane
                head_convs.append(ConvModule(inp, head_conv_plane, 3, padding=1))
        else:
            for i in range(conv_num):
                inp = self.planes[-1] if i == 0 else self.head_conv
                head_convs.append(ConvModule(inp, self.head_conv, 3, padding=1))

            if head_conv_plane:
                head_convs.append(ConvModule(self.head_conv, head_conv_plane, 3, padding=1))
            else:
                head_conv_plane = self.head_conv

        inp = self.planes[-1] if conv_num <= 0 else head_conv_plane
        head_convs.append(nn.Conv2d(inp, out_channel, 1))
        return nn.Sequential(*head_convs)

    def fill_up_weights(self, up):
        w = up.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

    def init_weights(self):
        for _, m in self.shortcut_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if self.use_deconv_init and isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.01)
            if self.use_trident and isinstance(m, nn.Conv2d):
                kaiming_init(m)

        if not self.predict_together:
            for _, m in self.hm.named_modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)

            if self.hm_init_value is not None:
                self.hm[-1].bias.data.fill_(self.hm_init_value)
            else:
                bias_cls = bias_init_with_prob(0.01)
                normal_init(self.hm[-1], std=0.01, bias=bias_cls)

            for _, m in self.wh.named_modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

            if self.wh_001:
                for _, m in self.wh.named_modules():
                    if isinstance(m, nn.Conv2d):
                        normal_init(m, std=0.01)
        else:
            for _, m in self.hmwh.named_modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

            bias_cls = bias_init_with_prob(0.01)
            self.hmwh[-1].bias.data[0:-1:5].fill_(bias_cls)

        if hasattr(self, 'centerness'):
            for _, m in self.centerness.named_modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 2, h, w).
            reg: None or tensor, (batch, 2, h, w).
        """
        x = feats[-1]
        for i, (deconv_layer, shortcut_layer) in enumerate(
                zip(self.deconv_layers, self.shortcut_layers)):
            x = deconv_layer(x)
            if self.use_shortcut:
                shortcut = shortcut_layer(feats[-i - 2])
                x = x + shortcut

        if not self.predict_together:
            hm = self.hm(x)
            wh = self.wh(x)
        else:
            N, _, H, W = x.shape
            hmwh = self.hmwh(x).view(N, -1, 5, H, W).transpose(1, 2)
            hm = hmwh[:, 0]
            wh = hmwh[:, 1:5].transpose(1, 2).contiguous().view(N, -1, H, W)
        wh = wh.exp() if self.use_exp_wh else F.relu(wh)

        if self.wh_offset_base is not None:
            if isinstance(self.wh_offset_base, nn.Module):
                wh = self.wh_offset_base(wh)
            else:
                wh *= self.wh_offset_base

        if self.norm_wh:
            N, _, H, W = wh.shape
            wh = wh.view(N, -1, 4, H, W).transpose(1, 2)
            wh[:, [0, 2]] = wh[:, [0, 2]] * hm.size(3)
            wh[:, [1, 3]] = wh[:, [1, 3]] * hm.size(2)
            wh = wh.transpose(1, 2).view(N, -1, H, W)

        if every_n_local_step(100):
            add_histogram_summary('ct_head_feat/heatmap', hm.detach().cpu())
            add_histogram_summary('ct_head_feat/wh', wh.detach().cpu())

            if not self.predict_together:
                hm_summary = self.hm[-1]
                wh_summary = self.wh[-1]
                add_histogram_summary('ct_head_param/hm', hm_summary.weight.detach().cpu(),
                                      is_param=True)
                add_histogram_summary('ct_head_param/wh', wh_summary.weight.detach().cpu(),
                                      is_param=True)

                add_histogram_summary('ct_head_param/hm_grad',
                                      hm_summary.weight.grad.detach().cpu(), collect_type='none')
                add_histogram_summary('ct_head_param/wh_grad',
                                      wh_summary.weight.grad.detach().cpu(), collect_type='none')
        centerness = None
        if self.fovea_hm and not self.only_merge:
            centerness = self.centerness(x)

        return hm, wh, centerness

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh', 'pred_centerness'))
    def get_bboxes(self,
                   pred_heatmap,
                   pred_wh,
                   pred_centerness,
                   img_metas,
                   cfg,
                   rescale=False):
        batch, cat, height, width = pred_heatmap.size()
        pred_heatmap = pred_heatmap.detach().sigmoid_()
        wh = pred_wh.detach()

        if self.fovea_hm and not self.only_merge:
            pred_centerness = pred_centerness.detach()
            pred_centerness = pred_centerness.sigmoid_()
            pred_heatmap = pred_heatmap * pred_centerness

        # perform nms on heatmaps
        heat = simple_nms(pred_heatmap)  # used maxpool to filter the max score

        topk = getattr(cfg, 'max_per_img', 100)
        # (batch, topk)
        scores, inds, clses, ys, xs = self._topk(heat, topk=topk)
        xs = xs.view(batch, topk, 1) * self.down_ratio
        ys = ys.view(batch, topk, 1) * self.down_ratio

        wh = tranpose_and_gather_feat(wh, inds)  # (batch, topk, 4) or (batch, topk, 80 * 4)
        if not self.wh_agnostic:
            wh = wh.view(-1, topk, self.num_fg, 4)
            wh = torch.gather(wh, 2, clses[..., None, None].expand(
                clses.size(0), clses.size(1), 1, 4).long())

        wh = wh.view(batch, topk, 4)
        clses = clses.view(batch, topk, 1).float()
        scores = scores.view(batch, topk, 1)

        bboxes = torch.cat([xs - wh[..., [0]],
                            ys - wh[..., [1]],
                            xs + wh[..., [2]],
                            ys + wh[..., [3]]], dim=2)

        result_list = []
        for idx in range(bboxes.shape[0]):
            scores_per_img = scores[idx]

            scores_keep = (scores_per_img > getattr(cfg, 'score_thr', 0.01)).squeeze(-1)
            scores_per_img = scores_per_img[scores_keep]
            bboxes_per_img = bboxes[idx][scores_keep]
            labels_per_img = clses[idx][scores_keep]

            if rescale:
                scale_factor = img_metas[idx]['scale_factor']
                bboxes_per_img /= bboxes_per_img.new_tensor(scale_factor)

            labels_per_img = labels_per_img.squeeze(-1)
            if self.use_nms:
                labels_per_img += 1
                bboxes_per_img, labels_per_img = nms_agnostic(
                    bboxes_per_img, scores_per_img.squeeze(-1), labels_per_img,
                    cfg.score_thr, cfg.nms, self.num_classes + 1, cfg.max_per_img)
            else:
                bboxes_per_img = torch.cat([bboxes_per_img, scores_per_img], dim=1)
            result_list.append((bboxes_per_img, labels_per_img))

        return result_list

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh', 'pred_centerness'))
    def loss(self,
             pred_heatmap,
             pred_wh,
             pred_centerness,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        all_targets = self._target_generator(gt_bboxes, gt_labels, img_metas)
        hm_loss, wh_loss, centerness_loss, merge_loss = self._loss(
            pred_heatmap, pred_wh, pred_centerness, *all_targets)
        return {'losses/centernext_loss_heatmap': hm_loss,
                'losses/centernext_loss_wh': wh_loss,
                'losses/centernext_loss_centerness': centerness_loss,
                'losses/centernext_loss_merge': merge_loss}

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


class CXTTargetGenerator(object):

    def __init__(self,
                 num_fg,
                 wh_planes,
                 fovea_hm,
                 avg_wh_weightv2,
                 avg_wh_weightv3,
                 avg_wh_weightv4,
                 only_merge,
                 wh_offset_base,
                 fill_small,
                 wh_area_process,
                 wh_heatmap,
                 hm_center_ratio,
                 ct_gaussian,
                 down_ratio=4,
                 max_objs=128,
                 center_ratio=0.2,
                 ignore_ratio=0.5,
                 wh_agnostic=False,
                 ct_version=True):
        self.num_fg = num_fg
        self.fovea_hm = fovea_hm
        self.wh_planes = wh_planes
        self.avg_wh_weightv2 = avg_wh_weightv2
        self.avg_wh_weightv3 = avg_wh_weightv3
        self.avg_wh_weightv4 = avg_wh_weightv4
        self.only_merge = only_merge
        self.wh_offset_base = wh_offset_base
        self.fill_small = fill_small
        self.wh_area_process = wh_area_process
        self.wh_heatmap = wh_heatmap
        self.hm_center_ratio = hm_center_ratio
        self.ct_gaussian = ct_gaussian
        self.down_ratio = down_ratio
        self.max_objs = max_objs
        self.center_ratio = center_ratio
        self.ignore_ratio = ignore_ratio
        self.wh_agnostic = wh_agnostic
        self.ct_version = ct_version

        if (self.center_ratio / 2 != self.hm_center_ratio) and self.wh_heatmap:
            assert not self.ct_gaussian

    def target_single_image(self, gt_boxes, gt_labels, feat_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w) or (80 * 4, h, w).
        """
        output_h, output_w = feat_shape
        heatmap_channel = self.num_fg

        heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        fake_heatmap = gt_boxes.new_zeros((output_h, output_w))
        box_target = gt_boxes.new_ones((self.wh_planes, output_h, output_w)) * -1
        wh_weight = gt_boxes.new_zeros((self.wh_planes // 4, output_h, output_w))
        hm_weight = gt_boxes.new_zeros((self.wh_planes // 4, output_h, output_w))
        centerness = gt_boxes.new_zeros((1, output_h, output_w))

        if self.wh_area_process == 'log':
            boxes_areas_log = bbox_areas(gt_boxes).log()
        elif self.wh_area_process == 'sqrt':
            boxes_areas_log = bbox_areas(gt_boxes).sqrt()
        else:
            boxes_areas_log = bbox_areas(gt_boxes)
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))

        if self.wh_area_process == 'norm':
            boxes_area_topk_log[:] = 1.

        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        feat_gt_boxes = gt_boxes / self.down_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                               max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                               max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        r1 = (1 - self.center_ratio) / 2
        r2 = (1 - self.ignore_ratio) / 2

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        if self.hm_center_ratio is None:
            radiuses = torch.clamp(gaussian_radius((feat_hs.ceil(), feat_ws.ceil())), min=0)
            hw_ratio_sqrt = (feat_hs / feat_ws).sqrt()
            h_radiuses = (radiuses * hw_ratio_sqrt).int()
            w_radiuses = (radiuses / hw_ratio_sqrt).int()
            if self.ct_gaussian:
                radiuses = radiuses.int()
        else:
            h_radiuses = (feat_hs * self.hm_center_ratio).int()
            w_radiuses = (feat_ws * self.hm_center_ratio).int()
            if (self.center_ratio / 2 != self.hm_center_ratio) and self.wh_heatmap:
                wh_h_radiuses = (feat_hs * self.center_ratio / 2).int()
                wh_w_radiuses = (feat_ws * self.center_ratio / 2).int()

        # calculate positive (center) regions
        ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = calc_region(gt_boxes.transpose(0, 1), r1,
                                                         use_round=False)
        ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = [torch.round(x / self.down_ratio).int()
                                              for x in [ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s]]
        ctr_x1s, ctr_x2s = [torch.clamp(x, max=output_w - 1) for x in [ctr_x1s, ctr_x2s]]
        ctr_y1s, ctr_y2s = [torch.clamp(y, max=output_h - 1) for y in [ctr_y1s, ctr_y2s]]
        ctr_xs_diff, ctr_ys_diff = ctr_x2s - ctr_x1s + 1, ctr_y2s - ctr_y1s + 1

        if self.fill_small:
            are_fill_small = (ctr_ys_diff <= 4) & (ctr_xs_diff <= 4)

        collide_pixels_summary = 0
        # larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k] - 1
            ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[k], ctr_y1s[k], ctr_x2s[k], ctr_y2s[k]
            ctr_x_diff, ctr_y_diff = ctr_xs_diff[k], ctr_ys_diff[k]

            if self.fovea_hm or (self.fill_small and are_fill_small[k]):
                ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(
                    feat_gt_boxes[k], r2, (output_h, output_w))

                if not self.fovea_hm:
                    ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ignore_x1, ignore_y1, ignore_x2, ignore_y2

            fake_heatmap = fake_heatmap.zero_()
            if self.ct_gaussian:
                draw_umich_gaussian(fake_heatmap, ct_ints[k], radiuses[k].item())
            else:
                draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                       h_radiuses[k].item(), w_radiuses[k].item())

            if self.fovea_hm:
                # ignore_mask_box is necessary to prevent the ignore areas covering the
                # pos areas of larger boxes
                ignore_mask_box = (
                        heatmap[
                        cls_id, ignore_y1:ignore_y2 + 1, ignore_x1:ignore_x2 + 1] == 0)
                heatmap[cls_id, ignore_y1:ignore_y2 + 1, ignore_x1:ignore_x2 + 1][
                    ignore_mask_box] = -1
                heatmap[cls_id, ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1
                centerness[0] = torch.max(centerness[0], fake_heatmap)
            else:
                heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)

            if self.wh_heatmap:
                if self.hm_center_ratio != self.center_ratio / 2:
                    fake_heatmap = fake_heatmap.zero_()
                    draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                           wh_h_radiuses[k].item(), wh_w_radiuses[k].item())
                box_target_inds = fake_heatmap > 0
            else:
                box_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1

            if self.wh_agnostic:
                collide_pixels_summary += (box_target[:, box_target_inds] > 0).sum()

                box_target[:, box_target_inds] = gt_boxes[k][:, None]
            else:
                collide_pixels_summary += (box_target[(cls_id * 4):(cls_id + 1) * 4,
                                           box_target_inds] > 0).sum()

                box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = gt_boxes[k][:, None]

            local_heatmap = fake_heatmap[box_target_inds]
            ct_div = local_heatmap.sum()
            local_heatmap *= boxes_area_topk_log[k]

            if self.wh_agnostic:
                cls_id = 0

            if self.avg_wh_weightv2 and ct_div > 0:
                wh_weight[cls_id, box_target_inds] = local_heatmap / ct_div
            elif self.avg_wh_weightv3 and ct_div > 0 and ctr_y_diff > 6 and ctr_x_diff > 6:
                wh_weight[cls_id, box_target_inds] = local_heatmap / ct_div
            elif self.avg_wh_weightv4 and ct_div > 0 and ctr_y_diff > 6 and ctr_x_diff > 6:
                wh_weight[cls_id, box_target_inds] = local_heatmap / ct_div
            else:
                wh_weight[cls_id, box_target_inds] = \
                    boxes_area_topk_log[k] / box_target_inds.sum().float()

            if self.avg_wh_weightv4:
                wh_weight[cls_id, ct_ints[k, 1].item(), ct_ints[k, 0].item()] = \
                    boxes_area_topk_log[k]

            if not self.ct_version:
                target_loc = fake_heatmap > 0.9
                hm_target_num = target_loc.sum().float()
                hm_weight[cls_id, target_loc] = 1 / (2 * (hm_target_num - 1))
                hm_weight[cls_id, ct_ints[k, 1].item(), ct_ints[k, 0].item()] = 1 / 2.

        add_summary('box_target', collide_pixels=collide_pixels_summary)
        pos_pixels_summary = (box_target > 0).sum()
        add_summary('box_target', pos_pixels=pos_pixels_summary)
        add_summary('box_target', collide_ratio=collide_pixels_summary / pos_pixels_summary.float())

        return heatmap, box_target, centerness, wh_weight, hm_weight

    def __call__(self, gt_boxes, gt_labels, img_metas):
        """

        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            centerness: tensor or None.
        """
        with torch.no_grad():
            feat_shape = (img_metas[0]['pad_shape'][0] // self.down_ratio,
                          img_metas[0]['pad_shape'][1] // self.down_ratio)
            heatmap, box_target, centerness, wh_weight, hm_weight = multi_apply(
                self.target_single_image,
                gt_boxes,
                gt_labels,
                feat_shape=feat_shape
            )

            heatmap, box_target = [torch.stack(t, dim=0).detach() for t in [heatmap, box_target]]
            if self.fovea_hm:
                centerness = torch.stack(centerness, dim=0).detach()

            wh_weight = torch.stack(wh_weight, dim=0).detach()

            if not self.ct_version:
                hm_weight = torch.stack(hm_weight, dim=0).detach()

            return heatmap, box_target, centerness, wh_weight, hm_weight


class CXTLoss(object):

    def __init__(self,
                 giou_weight=1.,
                 hm_weight=1.,
                 ct_weight=0.3,
                 merge_weight=0.3,
                 fovea_hm=False,
                 only_merge=False,
                 ct_version=True,
                 wh_agnostic=False,
                 down_ratio=4):
        super(CXTLoss, self).__init__()
        self.giou_weight = giou_weight
        self.hm_weight = hm_weight
        self.ct_weight = ct_weight
        self.merge_weight = merge_weight
        self.fovea_hm = fovea_hm
        self.only_merge = only_merge
        self.ct_version = ct_version
        self.wh_agnostic = wh_agnostic
        self.down_ratio = down_ratio

        self.base_loc = None

    def __call__(self,
                 pred_hm,
                 pred_wh,
                 pred_centerness,
                 heatmap,
                 box_target,
                 centerness,
                 wh_weight,
                 hm_weight):
        """

        Args:
            pred_hm: tensor, (batch, 80, h, w).
            pred_wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            pred_centerness: tensor or None, (batch, 1, h, w).
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            centerness: tensor or None, (batch, 1, h, w).
            wh_weight: tensor or None, (batch, 80, h, w).

        Returns:

        """
        if every_n_local_step(100):
            pred_hm_summary = torch.clamp(torch.sigmoid(pred_hm), min=1e-4, max=1 - 1e-4)
            gt_hm_summary = heatmap.clone()
            if self.fovea_hm:
                if not self.only_merge:
                    pred_ctn_summary = torch.clamp(torch.sigmoid(pred_centerness),
                                                   min=1e-4, max=1 - 1e-4)
                    add_feature_summary('centernet/centerness',
                                        pred_ctn_summary.detach().cpu().numpy(), type='f')
                    add_feature_summary(
                        'centernet/merge',
                        (pred_ctn_summary * pred_hm_summary).detach().cpu().numpy(), type='max')

                add_feature_summary('centernet/gt_centerness',
                                    centerness.detach().cpu().numpy(), type='f')
                add_feature_summary(
                    'centernet/gt_merge',
                    (centerness * gt_hm_summary).detach().cpu().numpy(), type='max')

            add_feature_summary('centernet/heatmap', pred_hm_summary.detach().cpu().numpy())
            add_feature_summary('centernet/gt_heatmap', gt_hm_summary.detach().cpu().numpy())

        H, W = pred_hm.shape[2:]
        if not self.fovea_hm:
            pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
            hm_weight = None if self.ct_version else hm_weight
            hm_loss = ct_focal_loss(pred_hm, heatmap, hm_weight=hm_weight) * self.hm_weight
            centerness_loss = hm_loss.new_tensor([0.])
            merge_loss = hm_loss.new_tensor([0.])
        else:
            care_mask = (heatmap >= 0).float()
            avg_factor = torch.sum(heatmap > 0).float().item() + 1e-6
            if not self.only_merge:
                hm_loss = py_sigmoid_focal_loss(
                    pred_hm, heatmap, care_mask, reduction='sum') / avg_factor * self.hm_weight

                pred_centerness = torch.clamp(torch.sigmoid(pred_centerness),
                                              min=1e-4, max=1 - 1e-4)
                centerness_loss = ct_focal_loss(
                    pred_centerness, centerness, gamma=2.) * self.ct_weight

                merge_loss = ct_focal_loss(torch.clamp(torch.sigmoid(pred_hm) * pred_centerness,
                                                       min=1e-4, max=1 - 1e-4),
                                           heatmap * centerness,
                                           weight=(heatmap >= 0).float()) * self.merge_weight
            else:
                hm_loss = pred_hm.new_tensor([0.])
                centerness_loss = pred_hm.new_tensor([0.])
                merge_loss = ct_focal_loss(torch.clamp(torch.sigmoid(pred_hm),
                                                       min=1e-4, max=1 - 1e-4),
                                           heatmap * centerness,
                                           weight=(heatmap >= 0).float()) * self.merge_weight

        if not self.wh_agnostic:
            pred_wh = pred_wh.view(pred_wh.size(0) * pred_hm.size(1), 4, H, W)
            box_target = box_target.view(box_target.size(0) * pred_hm.size(1), 4, H, W)
        mask = wh_weight.view(-1, H, W)
        avg_factor = mask.sum() + 1e-4

        if self.base_loc is None:
            base_step = self.down_ratio
            shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        # (batch, h, w, 4)
        pred_boxes = torch.cat((self.base_loc - pred_wh[:, [0, 1]],
                                self.base_loc + pred_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
        # (batch, h, w, 4)
        boxes = box_target.permute(0, 2, 3, 1)
        wh_loss = giou_loss(pred_boxes, boxes, mask, avg_factor=avg_factor) * self.giou_weight

        return hm_loss, wh_loss, centerness_loss, merge_loss
