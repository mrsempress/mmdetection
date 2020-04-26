import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import normal_init
from mmdet.core import multi_apply
from mmdet.core.utils.common import gather_feat, tranpose_and_gather_feat
from mmdet.core.utils.summary import (
    add_summary, every_n_local_step, add_feature_summary, add_histogram_summary)
from mmdet.models.losses import ct_focal_loss, weighted_l1, smooth_l1_loss, giou_loss
from mmdet.models.utils import (
    build_norm_layer, gaussian_radius, draw_umich_gaussian, ConvModule, simple_nms,
    build_conv_layer, ShortcutConv2d, draw_truncate_gaussian)
from mmdet.ops import ModulatedDeformConvPack, ModulatedDeformConv
from .anchor_head import AnchorHead
from ..registry import HEADS


@HEADS.register_module
class CTHead(AnchorHead):

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 head_conv=256,
                 hm_head_conv_num=1,
                 wh_head_conv_num=1,
                 deconv_with_bias=False,
                 num_classes=81,
                 use_reg_offset=True,
                 use_smooth_l1=False,
                 use_exp_wh=False,
                 use_giou=False,
                 use_shortcut=False,
                 use_upsample_conv=False,
                 use_trident=False,
                 use_dla=False,
                 shortcut_cfg=(1, 1, 1),
                 shortcut_attention=(False, False, False),
                 shortcut_kernel=3,
                 use_rep_points=False,
                 rep_points_kernel=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 neg_shortcut=False,
                 hm_init_value=-2.19,
                 use_exp_hm=False,
                 use_truncate_gaussia=False,
                 use_tight_gauusia=False,
                 gt_plus_dot5=False,
                 shortcut_in_shortcut=False,
                 giou_weight=1.,
                 wh_weight=0.1,
                 off_weight=1.,
                 hm_weight=1.):
        super(AnchorHead, self).__init__()
        assert len(planes) in [2, 3, 4] and \
               len(planes) == len(shortcut_cfg) == len(shortcut_attention)
        self.down_ratio = 32 // 2 ** len(planes)

        self.planes = planes
        self.head_conv = head_conv
        self.deconv_with_bias = deconv_with_bias
        self.num_classes = num_classes
        self.num_fg = num_classes - 1

        self.use_reg_offset = use_reg_offset
        self.use_shortcut = use_shortcut
        self.use_rep_points = use_rep_points
        assert rep_points_kernel in [3, 5]
        self.rep_points_kernel = rep_points_kernel
        self.use_exp_wh = use_exp_wh
        self.use_exp_hm = use_exp_hm
        self.use_upsample_conv = use_upsample_conv
        self.use_trident = use_trident
        self.use_dla = use_dla
        self.conv_cfg = conv_cfg
        self.neg_shortcut = neg_shortcut

        self.hm_init_value = hm_init_value
        self.wh_weight = wh_weight
        self.off_weight = off_weight
        self.hm_weight = hm_weight

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
            kernel_size=shortcut_kernel, padding=padding, shortcut_in_shortcut=shortcut_in_shortcut)

        # heads
        if use_rep_points:
            assert hm_head_conv_num == wh_head_conv_num
            share_head_conv = [ConvModule(self.planes[-1], head_conv, 3, padding=1)]
            for i in range(1, wh_head_conv_num):
                share_head_conv.append(ConvModule(head_conv, head_conv, 3, padding=1))
            self.share_head_conv = nn.Sequential(*share_head_conv)
            padding = 1 if rep_points_kernel == 3 else 2
            kernel_spatial = rep_points_kernel ** 2
            self.wh = nn.Conv2d(head_conv, kernel_spatial * 3, rep_points_kernel, padding=padding)
            self.hm = ModulatedDeformConv(head_conv, self.num_fg, rep_points_kernel,
                                          padding=padding)
        else:
            self.wh = self._make_conv_layer(2, wh_head_conv_num)
            self.hm = self._make_conv_layer(self.num_fg, hm_head_conv_num,
                                            use_exp_conv=use_exp_hm)
        if use_reg_offset:
            self.reg = self._make_conv_layer(2, wh_head_conv_num)

        self._target_generator = CTTargetGenerator(self.num_fg, use_reg_offset, use_giou,
                                                   use_truncate_gaussia, use_tight_gauusia,
                                                   gt_plus_dot5,
                                                   down_ratio=self.down_ratio)
        self._loss = CTLoss(use_reg_offset, use_smooth_l1, use_giou,
                            giou_weight, wh_weight, off_weight, hm_weight)

    def _make_shortcut(self,
                       inplanes,
                       planes,
                       shortcut_cfg,
                       shortcut_attention,
                       kernel_size=3,
                       padding=1,
                       shortcut_in_shortcut=False):
        assert len(inplanes) == len(planes) == len(shortcut_cfg) == len(shortcut_attention)

        shortcut_layers = nn.ModuleList()
        for (inp, outp, layer_num, attention) in zip(
                inplanes, planes, shortcut_cfg, shortcut_attention):
            if self.use_shortcut:
                assert layer_num > 0
                layer = ShortcutConv2d(
                    inp, outp, [kernel_size] * layer_num, [padding] * layer_num,
                    use_cbam=attention, shortcut_in_shortcut=shortcut_in_shortcut)
            else:
                layer = nn.Sequential()
            shortcut_layers.append(layer)
        return shortcut_layers

    def _get_deconv_cfg(self, deconv_kernel):
        assert deconv_kernel in [2, 3, 4], deconv_kernel
        if deconv_kernel == 4:
            padding, output_padding = (1, 0)
        elif deconv_kernel == 3:
            padding, output_padding = (1, 1)
        else:
            padding, output_padding = (0, 0)
        return deconv_kernel, padding, output_padding

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
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            inplanes = inplanes if i == 0 else num_filters[i - 1]

            if self.use_trident:
                mdcn = build_conv_layer(dict(type='TriConv'), inplanes, planes)
            else:
                mdcn = ModulatedDeformConvPack(inplanes, planes, 3, stride=1,
                                               padding=1, dilation=1, deformable_groups=1)
            if self.use_upsample_conv:
                up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                   nn.Conv2d(planes, planes, 3, padding=1))
            else:
                up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
                self.fill_up_weights(up)

            layers.append(mdcn)
            if norm_cfg:
                layers.append(build_norm_layer(norm_cfg, planes)[1])
            layers.append(nn.ReLU(inplace=True))

            layers.append(up)
            if norm_cfg:
                layers.append(build_norm_layer(norm_cfg, planes)[1])
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_conv_layer(self, out_channel, conv_num, use_exp_conv=False):
        head_convs = []
        for i in range(conv_num):
            inp = self.planes[-1] if i == 0 else self.head_conv
            head_convs.append(ConvModule(inp, self.head_conv, 3, padding=1))

        inp = self.planes[-1] if conv_num <= 0 else self.head_conv
        head_convs.append(nn.Conv2d(inp, out_channel, 1))
        if use_exp_conv:
            head_convs.append(build_conv_layer(
                dict(type='ExpConv'), out_channel, out_channel, neg_x=True))
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
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for _, m in self.shortcut_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        if not self.use_rep_points:
            if not self.use_exp_hm:
                self.hm[-1].bias.data.fill_(self.hm_init_value)
            else:
                self.hm[-1].conv.bias.data.fill_(self.hm_init_value)
        else:
            self.hm.bias.data.fill_(self.hm_init_value)

        for m in self.wh.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

        if self.use_reg_offset:
            for m in self.reg.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

        if self.use_rep_points:
            for _, m in self.share_head_conv.named_modules():
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
        if not self.use_dla:
            for i, (deconv_layer, shortcut_layer) in enumerate(
                    zip(self.deconv_layers, self.shortcut_layers)):
                x = deconv_layer(x)

                if self.use_shortcut:
                    shortcut = shortcut_layer(feats[-i - 2])
                    if self.neg_shortcut:
                        shortcut = -1 * F.relu(-1 * shortcut)
                    x = x + shortcut

                    if every_n_local_step(500):
                        add_feature_summary('ct_head_shortcut_{}'.format(i),
                                            shortcut.detach().cpu().numpy())

        if self.use_rep_points:
            share_feat = self.share_head_conv(x)
            o1, o2, mask = torch.chunk(self.wh(share_feat), 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)  # 18 channels for example, h1, w1, h2, w2, ...
            mask = torch.sigmoid(mask)
            hm = self.hm(share_feat, offset, mask)

            # seems like the code below will not improve the mAP, but it suppose to.
            kernel_spatial = self.rep_points_kernel ** 2
            o1, o2 = torch.chunk(offset.permute(0, 2, 3, 1).contiguous().view(
                -1, kernel_spatial, 2).transpose(1, 2).contiguous().view(
                offset.shape[0], *offset.shape[2:], kernel_spatial * 2).permute(
                0, 3, 1, 2), 2, dim=1)

            if every_n_local_step(100):
                for i in range(offset.shape[1]):
                    add_histogram_summary('ct_rep_points_{}'.format(i),
                                          offset[:, [i]].detach().cpu())

            radius = (self.rep_points_kernel - 1) // 2
            h_base = hm.new_tensor([i for i in range(-radius, radius + 1)])
            h_base = torch.stack([h_base for _ in range(self.rep_points_kernel)],
                                 dim=1).view(1, kernel_spatial, 1, 1)
            w_base = hm.new_tensor([i for i in range(-radius, radius + 1)])[None]
            w_base = torch.cat([w_base for _ in range(self.rep_points_kernel)],
                               dim=0).view(1, kernel_spatial, 1, 1)

            h_loc, w_loc = o1 + h_base, o2 + w_base
            wh = torch.cat([w_loc.max(1, keepdim=True)[0] - w_loc.min(1, keepdim=True)[0],
                            h_loc.max(1, keepdim=True)[0] - h_loc.min(1, keepdim=True)[0]], dim=1)
        else:
            hm = self.hm(x)
            wh = self.wh(x)
        reg = self.reg(x) if self.use_reg_offset else None
        if self.use_exp_wh:
            wh = wh.exp()

        if every_n_local_step(500):
            add_histogram_summary('ct_head_feat/heatmap', hm.detach().cpu())
            add_histogram_summary('ct_head_feat/wh', wh.detach().cpu())
            if self.use_reg_offset:
                add_histogram_summary('ct_head_feat/reg', reg.detach().cpu())

            if self.use_rep_points:
                hm_summary, wh_summary = self.hm, self.wh
            elif self.use_exp_hm:
                hm_summary, wh_summary = self.hm[-1].conv, self.wh[-1]
            else:
                hm_summary, wh_summary = self.hm[-1], self.wh[-1]

            add_histogram_summary('ct_head_param/hm', hm_summary.weight.detach().cpu(),
                                  is_param=True)
            add_histogram_summary('ct_head_param/wh', wh_summary.weight.detach().cpu(),
                                  is_param=True)
            if self.use_reg_offset:
                add_histogram_summary('ct_head_param/reg', self.reg[-1].weight.detach().cpu(),
                                      is_param=True)

            add_histogram_summary('ct_head_param/hm_grad',
                                  hm_summary.weight.grad.detach().cpu(), collect_type='none')
            add_histogram_summary('ct_head_param/wh_grad',
                                  wh_summary.weight.grad.detach().cpu(), collect_type='none')

        return hm, wh, reg

    def get_bboxes(self,
                   pred_heatmap,
                   pred_wh,
                   pred_reg_offset,
                   img_metas,
                   cfg,
                   rescale=False):
        batch, cat, height, width = pred_heatmap.size()
        pred_heatmap = pred_heatmap.detach()
        wh = pred_wh.detach()
        reg = pred_reg_offset.detach() if self.use_reg_offset else None

        # perform nms on heatmaps
        heat = simple_nms(pred_heatmap.sigmoid_())  # used maxpool to filter the max score

        topk = getattr(cfg, 'max_per_img', 100)
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
                            ys + wh[..., 1:2] / 2], dim=2)

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
            bboxes_per_img *= self.down_ratio

            bboxes_per_img = torch.cat([bboxes_per_img, scores_per_img], dim=1)
            labels_per_img = labels_per_img.squeeze(-1)
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
        return {'losses/centernet_loss_heatmap': hm_loss,
                'losses/centernet_loss_wh': wh_loss,
                'losses/centernet_loss_off': off_loss}

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
                 num_fg,
                 use_reg_offset,
                 use_giou,
                 use_truncate_gaussia,
                 use_tight_gauusia,
                 gt_plus_dot5,
                 down_ratio=4,
                 max_objs=128):
        self.num_fg = num_fg
        self.use_reg_offset = use_reg_offset
        self.use_giou = use_giou
        self.use_truncate_gaussia = use_truncate_gaussia
        self.use_tight_gauusia = use_tight_gauusia
        self.gt_plus_dot5 = gt_plus_dot5
        self.down_ratio = down_ratio
        self.max_objs = max_objs

    def target_single_image(self, gt_boxes, gt_labels, feat_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            wh: tensor, tensor <=> img, (max_obj, 2).
            reg_mask: tensor, tensor <=> img, (max_obj,).
            ind: tensor, tensor <=> img, (max_obj,).
            reg: tensor, tensor <=> img, (max_obj, 2).
            center_location: tensor or None, tensor <=> img, (max_obj, 2).
        """
        output_h, output_w = feat_shape
        heatmap = gt_boxes.new_zeros((self.num_fg, output_h, output_w))
        wh = gt_boxes.new_zeros((self.max_objs, 2))
        reg_mask = gt_boxes.new_zeros((self.max_objs,), dtype=torch.uint8)
        ind = gt_boxes.new_zeros((self.max_objs,), dtype=torch.long)

        reg, center_location = None, None
        if self.use_reg_offset:
            reg = gt_boxes.new_zeros((self.max_objs, 2))
        if self.use_giou:
            center_location = gt_boxes.new_zeros((self.max_objs, 2))

        gt_boxes /= self.down_ratio
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
                if self.gt_plus_dot5:
                    ct_int = (center + 0.5).to(torch.int)
                else:
                    ct_int = center.to(torch.int)
                if self.use_truncate_gaussia:
                    if self.use_tight_gauusia:
                        h_radius = (h / 2).int().item()
                        w_radius = (w / 2).int().item()
                    else:
                        radius = gaussian_radius((h.ceil(), w.ceil()))
                        radius = max(0, int(radius.item()))
                        h_radius = (radius * (h / w).sqrt()).int().item()
                        w_radius = (radius * (w / h).sqrt()).int().item()
                    draw_truncate_gaussian(heatmap[cls_id], ct_int, h_radius, w_radius)
                else:
                    radius = gaussian_radius((h.ceil(), w.ceil()))
                    radius = max(0, int(radius.item()))
                    draw_umich_gaussian(heatmap[cls_id], ct_int, radius)
                # directly predict the width and height
                wh[k] = wh.new_tensor([1. * w, 1. * h])
                ind[k] = ct_int[1] * output_w + ct_int[0]
                if self.use_reg_offset:
                    reg[k] = center - ct_int.float()
                if self.use_giou:
                    center_location[k] = center
                reg_mask[k] = 1

        return heatmap, wh, reg_mask, ind, reg, center_location

    def __call__(self, gt_boxes, gt_labels, img_metas):
        """

        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            wh: tensor, (batch, max_obj, 2).
            reg_mask: tensor, tensor <=> img, (batch, max_obj).
            ind: tensor, (batch, max_obj).
            reg: tensor, (batch, max_obj, 2).
            center_location: tensor or None, (batch, max_obj, 2).
        """
        feat_shape = (img_metas[0]['pad_shape'][0] // self.down_ratio,
                      img_metas[0]['pad_shape'][1] // self.down_ratio)
        heatmap, wh, reg_mask, ind, reg, center_location = multi_apply(
            self.target_single_image,
            gt_boxes,
            gt_labels,
            feat_shape=feat_shape
        )

        with torch.no_grad():
            heatmap, wh, ind, reg_mask = [torch.stack(t, dim=0).detach()
                                          for t in [heatmap, wh, ind, reg_mask]]
            if self.use_reg_offset:
                reg = torch.stack(reg, dim=0).detach()
            if self.use_giou:
                center_location = torch.stack(center_location, dim=0).detach()

            return heatmap, wh, reg_mask, ind, reg, center_location


class CTLoss(object):

    def __init__(self,
                 use_reg_offset=True,
                 use_smooth_l1=False,
                 use_giou=False,
                 giou_weight=1.,
                 wh_weight=0.1,
                 off_weight=1.,
                 hm_weight=1.):
        super(CTLoss, self).__init__()
        self.use_reg_offset = use_reg_offset
        self.use_smooth_l1 = use_smooth_l1
        self.use_giou = use_giou
        self.giou_weight = giou_weight
        self.wh_weight = wh_weight
        self.off_weight = off_weight
        self.hm_weight = hm_weight

    def __call__(self,
                 pred_hm,
                 pred_wh,
                 pred_reg_offset,
                 heatmap,
                 wh,
                 reg_mask,
                 ind,
                 reg_offset,
                 center_location):
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
            center_location: tensor, (batch, max_obj, 2). Only useful when using GIOU.

        Returns:

        """
        H, W = pred_hm.shape[2:]
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        hm_loss = ct_focal_loss(pred_hm, heatmap) * self.hm_weight

        # (batch, 2, h, w) => (batch, max_obj, 2)
        pred = tranpose_and_gather_feat(pred_wh, ind)
        mask = reg_mask.unsqueeze(2).expand_as(pred).float()
        avg_factor = mask.sum() + 1e-4

        if self.use_giou:
            pred_boxes = torch.cat(
                (center_location - pred / 2., center_location + pred / 2.), dim=2)
            box_br = center_location + wh / 2.
            box_br[:, :, 0] = box_br[:, :, 0].clamp(max=W - 1)
            box_br[:, :, 1] = box_br[:, :, 1].clamp(max=H - 1)
            boxes = torch.cat((torch.clamp(center_location - wh / 2., min=0), box_br), dim=2)
            mask_no_expand = mask[:, :, 0]
            wh_loss = giou_loss(pred_boxes, boxes, mask_no_expand) * self.giou_weight
        else:
            if self.use_smooth_l1:
                wh_loss = smooth_l1_loss(pred, wh, mask, avg_factor=avg_factor) * self.wh_weight
            else:
                wh_loss = weighted_l1(pred, wh, mask, avg_factor=avg_factor) * self.wh_weight

        off_loss = hm_loss.new_tensor(0.)
        if self.use_reg_offset:
            pred_reg = tranpose_and_gather_feat(pred_reg_offset, ind)
            off_loss = weighted_l1(pred_reg, reg_offset, mask,
                                   avg_factor=avg_factor) * self.off_weight

            add_summary('centernet', gt_reg_off=reg_offset[reg_offset > 0].mean().item())

        if every_n_local_step(500):
            add_feature_summary('centernet/heatmap', pred_hm.detach().cpu().numpy())
            add_feature_summary('centernet/gt_heatmap', heatmap.detach().cpu().numpy())
            if self.use_reg_offset:
                add_feature_summary('centernet/reg_offset', pred_reg_offset.detach().cpu().numpy())

        return hm_loss, wh_loss, off_loss
