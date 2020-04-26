import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init
import numpy as np

from mmdet.ops import ModulatedDeformConvPack, RoIAlign, soft_nms
from mmdet.core import multi_apply, bbox_areas, force_fp32
from mmdet.core.utils.summary import write_txt
from mmdet.core.anchor.guided_anchor_target import calc_region
from mmdet.models.losses import ct_focal_loss, giou_loss, diou_loss, ciou_loss
from mmdet.models.utils import (build_norm_layer, bias_init_with_prob, ConvModule,
                                simple_nms, build_conv_layer, SEBlock)
from .anchor_head import AnchorHead
from ..registry import HEADS


@HEADS.register_module
class TTFHead(AnchorHead):

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 down_ratio=4,
                 head_conv=256,
                 wh_conv=64,
                 hm_head_conv_num=2,
                 wh_head_conv_num=2,
                 num_classes=81,
                 shortcut_kernel=3,
                 conv_cfg=None,
                 head_conv_size=3,
                 use_trident=False,
                 use_dla=False,
                 wh_sym=False,
                 upsample_vanilla_conv=False,
                 upsample_multiscale_conv=False,
                 up_conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 shortcut_cfg=(1, 2, 3),
                 wh_offset_base=16.,
                 wh_area_process='log',
                 wh_agnostic=True,
                 wh_gaussian=True,
                 box_size_range=None,
                 two_stage=False,
                 alpha=0.54,
                 beta=0.54,
                 hm_weight=1.,
                 dcn_mean=False,
                 iou_type='giou',
                 use_simple_nms=True,
                 aug_reg=False,
                 hm_last_3x3=False,
                 hm_last_3x3_d2=False,
                 hm_last_se3x3=False,
                 hm_last_5x5=False,
                 hm_last_7x7=False,
                 no_wh_se=False,
                 wh_weight=5.,
                 max_objs=128):
        super(AnchorHead, self).__init__()
        assert len(planes) in [2, 3, 4]
        shortcut_num = min(len(inplanes) - 1, len(planes))
        assert shortcut_num == len(shortcut_cfg)
        assert wh_area_process in [None, 'norm', 'log', 'sqrt']

        self.planes = planes
        self.head_conv = head_conv
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.head_conv_size = head_conv_size
        self.use_trident = use_trident
        self.use_dla = use_dla
        self.wh_sym = wh_sym
        self.upsample_vanilla_conv = upsample_vanilla_conv
        self.upsample_multiscale_conv = upsample_multiscale_conv
        self.up_conv_cfg = up_conv_cfg
        self.wh_offset_base = wh_offset_base
        self.wh_area_process = wh_area_process
        self.wh_agnostic = wh_agnostic
        self.wh_gaussian = wh_gaussian
        self.box_size_range = box_size_range
        self.two_stage = two_stage
        self.alpha = alpha
        self.beta = beta
        self.hm_weight = hm_weight
        self.dcn_mean = dcn_mean
        self.iou_loss = eval(iou_type + '_loss')
        self.use_simple_nms = use_simple_nms
        self.aug_reg = aug_reg
        self.hm_last_3x3 = hm_last_3x3
        self.hm_last_3x3_d2 = hm_last_3x3_d2
        self.hm_last_se3x3 = hm_last_se3x3
        self.no_wh_se = no_wh_se
        self.hm_last_5x5 = hm_last_5x5
        self.hm_last_7x7 = hm_last_7x7
        self.wh_weight = wh_weight
        self.max_objs = max_objs
        self.fp16_enabled = False

        self.down_ratio = down_ratio
        self.num_fg = num_classes - 1
        self.wh_planes = 4 if wh_agnostic else 4 * self.num_fg
        self.base_loc = None

        # repeat upsampling n times. 32x to 4x by default.
        self.deconv_layers = nn.ModuleList([
            self.build_upsample(inplanes[-1], planes[0], norm_cfg=norm_cfg),
            self.build_upsample(planes[0], planes[1], norm_cfg=norm_cfg)
        ])
        for i in range(2, len(planes)):
            self.deconv_layers.append(
                self.build_upsample(planes[i - 1], planes[i],
                                    norm_cfg=norm_cfg, no_upsample=(down_ratio == 8)))

        padding = (shortcut_kernel - 1) // 2
        self.shortcut_layers = self.build_shortcut(
            inplanes[:-1][::-1][:shortcut_num], planes[:shortcut_num], shortcut_cfg,
            kernel_size=shortcut_kernel, padding=padding)

        # heads
        self.wh = self.build_head(self.wh_planes, wh_head_conv_num,
                                  head_conv_plane=wh_conv, use_sym_conv=wh_sym)
        self.hm = self.build_head(self.num_fg, hm_head_conv_num)
        if two_stage:
            assert wh_agnostic
            self.align = RoIAlign(7, spatial_scale=1 / 4., sample_num=2)
            self.wh2 = nn.Sequential(ConvModule(self.planes[-1], 32, 5, norm_cfg=norm_cfg),  # 3x3
                                     ConvModule(32, 32, 3, norm_cfg=norm_cfg),
                                     ConvModule(32, 32, 1),
                                     ConvModule(32, 4, 1, activation=None))

    def build_shortcut(self,
                       inplanes,
                       planes,
                       shortcut_cfg,
                       kernel_size=3,
                       padding=1):
        assert len(inplanes) == len(planes) == len(shortcut_cfg)

        shortcut_layers = nn.ModuleList()
        for i, (inp, outp, layer_num) in enumerate(zip(
                inplanes, planes, shortcut_cfg)):
            assert layer_num > 0
            layer = ShortcutConv2d(
                inp, outp, [kernel_size] * layer_num, [padding] * layer_num,
                down=(self.down_ratio == 8 and i == len(inplanes) - 1))
            shortcut_layers.append(layer)
        return shortcut_layers

    def build_upsample(self, inplanes, planes, norm_cfg=None, no_upsample=False):
        if self.upsample_vanilla_conv:
            if isinstance(self.upsample_vanilla_conv, int):
                padding = int((self.upsample_vanilla_conv - 1) / 2)
                dila = padding
                mdcn = nn.Conv2d(inplanes, planes, 3, stride=1, padding=padding, dilation=dila)
            else:
                mdcn = nn.Conv2d(inplanes, planes, 3, stride=1, padding=1)
        elif self.upsample_multiscale_conv:
            mdcn = build_conv_layer(dict(type='MultiScaleConv'), inplanes, planes)
        elif self.use_trident:
            mdcn = build_conv_layer(dict(type='TriConv'), inplanes, planes)
        elif self.up_conv_cfg:
            mdcn = build_conv_layer(self.up_conv_cfg, inplanes, planes)
        else:
            mdcn = ModulatedDeformConvPack(inplanes, planes, 3, offset_mean=self.dcn_mean, stride=1,
                                           padding=1, dilation=1, deformable_groups=1)
        layers = []
        layers.append(mdcn)
        if norm_cfg:
            layers.append(build_norm_layer(norm_cfg, planes)[1])
        layers.append(nn.ReLU(inplace=True))
        if not no_upsample:
            up = nn.UpsamplingBilinear2d(scale_factor=2)
            layers.append(up)

        return nn.Sequential(*layers)

    def build_head(self, out_channel, conv_num=1, head_conv_plane=None, use_sym_conv=False):
        head_convs = []
        head_conv_plane = self.head_conv if not head_conv_plane else head_conv_plane
        for i in range(conv_num):
            inp = self.planes[-1] if i == 0 else head_conv_plane
            head_convs.append(ConvModule(inp, head_conv_plane,
                                         self.head_conv_size, conv_cfg=self.conv_cfg, padding=1))

        inp = self.planes[-1] if conv_num <= 0 else head_conv_plane
        if use_sym_conv:
            assert out_channel == 4
            head_convs.append(nn.Conv2d(inp, out_channel, 3, padding=1))
            # head_convs.append(ConvModule(inp, out_channel, 3, conv_cfg=dict(type='WHSymConv')))
        else:
            if self.hm_last_3x3:
                head_convs.append(nn.Conv2d(inp, out_channel, 3, padding=1))
            elif self.hm_last_3x3_d2:
                head_convs.append(nn.Conv2d(inp, out_channel, 3, padding=2, dilation=2))
            elif self.hm_last_5x5:
                head_convs.append(nn.Conv2d(inp, out_channel, 5, padding=2))
            elif self.hm_last_7x7:
                head_convs.append(nn.Conv2d(inp, out_channel, 7, padding=3))
            elif self.hm_last_se3x3:
                head_convs.append(nn.Conv2d(inp, out_channel, 3, padding=1))
                if not self.no_wh_se or out_channel != 4:
                    head_convs.append(SEBlock(out_channel, compress_ratio=4))
            else:
                head_convs.append(nn.Conv2d(inp, out_channel, 1))
        return nn.Sequential(*head_convs)

    def init_weights(self):
        for _, m in self.shortcut_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for _, m in self.hm.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        if self.hm_last_se3x3:
            normal_init(self.hm[-2], std=0.01, bias=bias_cls)
        else:
            normal_init(self.hm[-1], std=0.01, bias=bias_cls)

        for _, m in self.wh.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

        if self.two_stage:
            for _, m in self.wh2.named_modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        x = feats[-1]
        if not self.use_dla:
            for i, upsample_layer in enumerate(self.deconv_layers):
                x = upsample_layer(x)
                if i < len(self.shortcut_layers):
                    shortcut = self.shortcut_layers[i](feats[-i - 2])
                    x = x + shortcut
        hm = self.hm(x)
        wh = F.relu(self.wh(x)) * self.wh_offset_base
        return x, hm, wh

    @force_fp32(apply_to=('pred_feat', 'pred_heatmap', 'pred_wh'))
    def get_bboxes(self,
                   pred_feat,
                   pred_heatmap,
                   pred_wh,
                   img_metas,
                   cfg,
                   rescale=False):
        batch, cat, height, width = pred_heatmap.size()
        pred_heatmap = pred_heatmap.detach().sigmoid_()
        wh = pred_wh.detach()
        # write_txt(pred_heatmap, filename='pred_hm', thre=0.001)
        # perform nms on heatmaps
        if self.use_simple_nms and not getattr(cfg, 'debug', False):
            heat = simple_nms(pred_heatmap)  # used maxpool to filter the max score
        else:
            heat = pred_heatmap
            kernel = 3
            pad = (kernel - 1) // 2
            hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
            keep = (hmax == heat).float()
            keep_pad = keep.new_zeros(batch, cat, height + 2, width + 2)
            keep_pad[..., 1:-1, 1:-1] = keep
            keep = keep_pad
            # keep = ((keep[..., :-2, :-2] + keep[..., :-2, 1:-1] + keep[..., :-2, 2:] +
            #          keep[..., 1:-1, :-2] + keep[..., 1:-1, 1:-1] + keep[..., 1:-1, 2:] +
            #          keep[..., 2:, :-2] + keep[..., 2:, 1:-1] + keep[..., 2:, 2:]) > 0).float()
            keep = ((keep[..., :-2, 1:-1] +
                     keep[..., 1:-1, :-2] + keep[..., 1:-1, 1:-1] + keep[..., 1:-1, 2:] +
                     keep[..., 2:, 1:-1]) > 0).float()
            heat = heat * keep

        topk = getattr(cfg, 'max_per_img', 100)
        # (batch, topk)
        scores, inds, clses, ys, xs = self._topk(heat, topk=topk)
        xs = xs.view(batch, topk, -1, 1) * self.down_ratio
        ys = ys.view(batch, topk, -1, 1) * self.down_ratio

        wh = wh.permute(0, 2, 3, 1).contiguous()
        wh = wh.view(wh.size(0), -1, wh.size(3))
        inds = inds.view(batch, -1, 1)
        wh_inds = inds.expand(*inds.shape[:-1], wh.size(2))
        wh = wh.gather(1, wh_inds)

        if not self.wh_agnostic:
            wh = wh.view(-1, topk, self.num_fg, 4)
            wh = torch.gather(wh, 2, clses[..., None, None].expand(
                clses.size(0), clses.size(1), 1, 4).long())

        wh = wh.view(batch, topk, -1, 4)
        clses = clses.view(batch, topk, 1).long()
        scores = scores.view(batch, topk, 1)

        bboxes = torch.cat([xs - wh[..., [0]], ys - wh[..., [1]],
                            xs + wh[..., [2]], ys + wh[..., [3]]], dim=-1)
        if self.aug_reg:
            heat = pred_heatmap.permute(0, 2, 3, 1).contiguous()
            heat = heat.view(heat.size(0), -1, heat.size(3))
            score_inds = inds.expand(*inds.shape[:-1], heat.size(2))
            area_scores = heat.gather(1, score_inds).view(batch, topk, -1, self.num_fg)
            area_scores = area_scores.gather(-1, clses.expand(
                *clses.shape[:-1], area_scores.size(-2)).unsqueeze(-1)).squeeze(-1)

            bbox_weight = torch.cat([bboxes.new_ones((*bboxes.shape[:-2], 1)),
                                     torch.exp(-1 / (2 * (wh[..., 0, :] / 24) ** 2))],
                                    dim=-1) * area_scores
            # print(bbox_weight)
            bboxes = (bboxes * bbox_weight.unsqueeze(-1)).sum(-2) / bbox_weight.sum(-1,
                                                                                    keepdims=True)
        else:
            bboxes = bboxes.squeeze(-2)

        clses = clses.float()
        roi_boxes = bboxes.new_tensor([])
        if self.two_stage:
            for batch_i in range(bboxes.shape[0]):
                vaid_pre_boxes_i = bboxes[batch_i]  # (xx, 4)
                roi_boxes = torch.cat([
                    roi_boxes, torch.cat([
                        vaid_pre_boxes_i.new_ones([vaid_pre_boxes_i.size(0), 1]) * batch_i,
                        vaid_pre_boxes_i], dim=1)], dim=0)

            if roi_boxes.size(0) > 0:
                rois = self.align(pred_feat, roi_boxes)  # (n, cha, 7, 7)
                pred_wh2 = self.wh2(rois).view(-1, 4)
                bboxes = bboxes.view(-1, 4)
                bboxes[:, [0, 1]] = bboxes[:, [0, 1]] - pred_wh2[:, [0, 1]] * 16
                bboxes[:, [2, 3]] = bboxes[:, [2, 3]] + pred_wh2[:, [2, 3]] * 16
                bboxes = bboxes.view(batch, topk, 4)

        result_list = []
        score_thr = getattr(cfg, 'score_thr', 0.01)
        for batch_i in range(bboxes.shape[0]):
            scores_per_img = scores[batch_i]
            scores_keep = (scores_per_img > score_thr).squeeze(-1)

            scores_per_img = scores_per_img[scores_keep]
            bboxes_per_img = bboxes[batch_i][scores_keep]
            labels_per_img = clses[batch_i][scores_keep].squeeze(-1)
            img_shape = img_metas[batch_i]['pad_shape']
            bboxes_per_img[:, 0::2] = bboxes_per_img[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
            bboxes_per_img[:, 1::2] = bboxes_per_img[:, 1::2].clamp(min=0, max=img_shape[0] - 1)

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

            result_list.append((bboxes_per_img, labels_per_img))

        return result_list

    @force_fp32(apply_to=('pred_feat', 'pred_heatmap', 'pred_wh'))
    def loss(self,
             pred_feat,
             pred_heatmap,
             pred_wh,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        all_targets = self.target_generator(gt_bboxes, gt_labels, img_metas)
        hm_loss, wh_loss, wh2_loss = self.loss_calc(pred_feat, pred_heatmap, pred_wh, *all_targets)
        return {'losses/ttfnet_loss_heatmap': hm_loss, 'losses/ttfnet_loss_wh': wh_loss,
                'losses/ttfnet_loss_wh2': wh2_loss}

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
        topk_ind = topk_ind.unsqueeze(2)
        topk_inds = topk_inds.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_ys = topk_ys.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_xs = topk_xs.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)

        if self.aug_reg:
            expand_topk_inds = topk_inds.unsqueeze(-1).expand(*topk_inds.shape, 5)
            expand_topk_ys = topk_ys.unsqueeze(-1).expand(*topk_ys.shape, 5)
            expand_topk_xs = topk_xs.unsqueeze(-1).expand(*topk_xs.shape, 5)
            topk_inds = torch.stack((topk_inds, topk_inds - 1, topk_inds - width,
                                     topk_inds + 1, topk_inds + width), dim=2)
            topk_ys = torch.stack((topk_ys, topk_ys, topk_ys - 1, topk_ys, topk_ys + 1), dim=2)
            topk_xs = torch.stack((topk_xs, topk_xs - 1, topk_xs, topk_xs + 1, topk_xs), dim=2)
            aug_err_ys = (topk_ys >= height) | (topk_ys < 0)
            aug_err_xs = (topk_xs >= width) | (topk_xs < 0)
            aug_err_inds = (topk_inds >= (height * width)) | (topk_inds < 0)
            aug_err = aug_err_ys | aug_err_xs | aug_err_inds
            topk_ys[aug_err] = expand_topk_ys[aug_err]
            topk_xs[aug_err] = expand_topk_xs[aug_err]
            topk_inds[aug_err] = expand_topk_inds[aug_err]

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
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
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                          w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def target_single_image(self, gt_boxes, gt_labels, feat_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w) or (80 * 4, h, w).
            reg_weight: tensor, same as box_target
        """
        output_h, output_w = feat_shape
        heatmap_channel = self.num_fg

        heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        fake_heatmap = gt_boxes.new_zeros((output_h, output_w))
        box_target = gt_boxes.new_ones((self.wh_planes, output_h, output_w)) * -1
        reg_weight = gt_boxes.new_zeros((self.wh_planes // 4, output_h, output_w))

        boxes_areas_log = bbox_areas(gt_boxes)
        if self.box_size_range:
            keep_idx = (self.box_size_range[1] ** 2 >= boxes_areas_log) &\
                       (boxes_areas_log >= self.box_size_range[0] ** 2)
            boxes_areas_log = boxes_areas_log[keep_idx]
            gt_boxes = gt_boxes[keep_idx]
            gt_labels = gt_labels[keep_idx]
        if self.wh_area_process == 'log':
            boxes_areas_log = boxes_areas_log.log()
        elif self.wh_area_process == 'sqrt':
            boxes_areas_log = boxes_areas_log.sqrt()
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

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
        if self.wh_gaussian and self.alpha != self.beta:
            h_radiuses_beta = (feat_hs / 2. * self.beta).int()
            w_radiuses_beta = (feat_ws / 2. * self.beta).int()

        if not self.wh_gaussian:
            # calculate positive (center) regions
            r1 = (1 - self.beta) / 2
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = calc_region(gt_boxes.transpose(0, 1), r1)
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = [torch.round(x.float() / self.down_ratio).int()
                                                  for x in [ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s]]
            ctr_x1s, ctr_x2s = [torch.clamp(x, max=output_w - 1) for x in [ctr_x1s, ctr_x2s]]
            ctr_y1s, ctr_y2s = [torch.clamp(y, max=output_h - 1) for y in [ctr_y1s, ctr_y2s]]

        # larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k] - 1

            fake_heatmap = fake_heatmap.zero_()
            self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item(), w_radiuses_alpha[k].item())
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)

            if self.wh_gaussian:
                if self.alpha != self.beta:
                    fake_heatmap = fake_heatmap.zero_()
                    self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                                h_radiuses_beta[k].item(),
                                                w_radiuses_beta[k].item())
                box_target_inds = fake_heatmap > 0
            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[k], ctr_y1s[k], ctr_x2s[k], ctr_y2s[k]
                box_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1

            if self.wh_agnostic:
                box_target[:, box_target_inds] = gt_boxes[k][:, None]
                cls_id = 0
            else:
                box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = gt_boxes[k][:, None]

            if self.wh_gaussian:
                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
            else:
                reg_weight[cls_id, box_target_inds] = \
                    boxes_area_topk_log[k] / box_target_inds.sum().float()

            reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div

        return heatmap, box_target, reg_weight

    def target_generator(self, gt_boxes, gt_labels, img_metas):
        """

        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            reg_weight: tensor, same as box_target.
        """
        with torch.no_grad():
            feat_shape = (img_metas[0]['pad_shape'][0] // self.down_ratio,
                          img_metas[0]['pad_shape'][1] // self.down_ratio)
            heatmap, box_target, reg_weight = multi_apply(
                self.target_single_image,
                gt_boxes,
                gt_labels,
                feat_shape=feat_shape
            )

            heatmap, box_target = [torch.stack(t, dim=0).detach() for t in [heatmap, box_target]]
            reg_weight = torch.stack(reg_weight, dim=0).detach()

            return heatmap, box_target, reg_weight

    def loss_calc(self,
                  pred_feat,
                  pred_hm,
                  pred_wh,
                  heatmap,
                  box_target,
                  wh_weight):
        """

        Args:
            pred_hm: tensor, (batch, 80, h, w).
            pred_wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            heatmap: tensor, same as pred_hm.
            box_target: tensor, same as pred_wh.
            wh_weight: tensor, same as pred_wh.

        Returns:
            hm_loss
            wh_loss
        """
        H, W = pred_hm.shape[2:]
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        hm_loss = ct_focal_loss(pred_hm, heatmap) * self.hm_weight

        mask = wh_weight.view(-1, H, W)
        avg_factor = mask.sum() + 1e-4

        if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
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
        wh_loss = self.iou_loss(pred_boxes, boxes, mask,
                                avg_factor=avg_factor) * self.wh_weight

        wh2_loss = wh_loss.new_zeros([1])
        if self.two_stage:
            heat = simple_nms(pred_hm)
            scores, inds, clses, ys, xs = self._topk(heat, topk=100)

            pred_boxes_2 = pred_boxes.view(pred_boxes.size(0), -1, pred_boxes.size(3))
            boxes_2 = boxes.view(*pred_boxes_2.shape)
            inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), pred_boxes_2.size(2))
            pred_boxes_2 = pred_boxes_2.gather(1, inds)  # (batch, 100, 4)
            boxes_2 = boxes_2.gather(1, inds)

            score_thr = 0.01
            scores_keep = scores > score_thr  # (batch, topk)

            batch_idx = pred_boxes_2.new_tensor(torch.arange(0., pred_boxes_2.shape[0], 1.)).view(
                -1, 1, 1).expand(pred_boxes_2.shape[0], pred_boxes_2.shape[1], 1)[scores_keep]
            pred_boxes_2 = pred_boxes_2[scores_keep]
            boxes_2 = boxes_2[scores_keep].detach()

            valid_boxes = (boxes_2 >= 0).min(1)[0]
            batch_idx = batch_idx[valid_boxes]  # (n, 1)
            pred_boxes_2 = pred_boxes_2[valid_boxes]  # (n, 4)
            boxes_2 = boxes_2[valid_boxes]  # (n, 4)
            roi_boxes = torch.cat((batch_idx, pred_boxes_2), dim=1).detach()

            if roi_boxes.size(0) > 0:
                rois = self.align(pred_feat, roi_boxes)  # (n, cha, 7, 7)
                pred_wh2 = self.wh2(rois).view(-1, 4)
                pred_boxes_2[:, [0, 1]] = pred_boxes_2[:, [0, 1]].detach() - \
                                          pred_wh2[:, [0, 1]] * 16
                pred_boxes_2[:, [2, 3]] = pred_boxes_2[:, [2, 3]].detach() + \
                                          pred_wh2[:, [2, 3]] * 16
                wh2_loss = giou_loss(pred_boxes_2, boxes_2,
                                     boxes_2.new_ones(boxes_2.size(0)))

        return hm_loss, wh_loss, wh2_loss


class ShortcutConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 paddings,
                 activation_last=False,
                 down=False):
        super(ShortcutConv2d, self).__init__()
        assert len(kernel_sizes) == len(paddings)

        layers = []
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes, paddings)):
            inc = in_channels if i == 0 else out_channels
            if i == 0 and down:
                layers.append(nn.Conv2d(inc, out_channels, kernel_size,
                                        padding=padding, stride=2))
            else:
                layers.append(nn.Conv2d(inc, out_channels, kernel_size, padding=padding))
            if i < len(kernel_sizes) - 1 or activation_last:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.cat([x[0], F.upsample(x[1], scale_factor=2)], dim=1)
        y = self.layers(x)
        return y
