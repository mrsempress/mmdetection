import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from mmcv.cnn import normal_init
from mmdet.core import multi_apply
from mmdet.core.utils.common import gather_feat, tranpose_and_gather_feat
from mmdet.core.utils.summary import (add_summary, every_n_local_step,
                                      add_feature_summary,
                                      add_histogram_summary)
from mmdet.models.losses import (ct_focal_loss, weighted_l1, smooth_l1_loss,
                                 giou_loss, cross_entropy)
from mmdet.models.utils import (build_norm_layer, gaussian_radius,
                                draw_umich_gaussian, ConvModule, simple_nms,
                                build_conv_layer, ShortcutConv2d,
                                draw_truncate_gaussian)
from mmdet.ops import ModulatedDeformConvPack, ModulatedDeformConv
from .anchor_head import AnchorHead
from ..registry import HEADS


@HEADS.register_module
class CTBiHead(AnchorHead):

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 head_conv=256,
                 hm_head_conv_num=1,
                 wh_head_conv_num=1,
                 deconv_with_bias=False,
                 num_classes=81,
                 use_smooth_l1=False,
                 use_exp_wh=False,
                 use_shortcut=False,
                 use_upsample_conv=False,
                 use_trident=True,
                 shortcut_cfg=(1, 1, 1),
                 shortcut_attention=(False, False, False),
                 shortcut_kernel=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 neg_shortcut=False,
                 hm_init_value=-2.19,
                 use_exp_hm=False,
                 use_truncate_gaussia=False,
                 use_tight_gauusia=False,
                 gt_plus_dot5=False,
                 shortcut_in_shortcut=False,
                 heights_weight=0.5,
                 xoff_weight=0.5,
                 yoff_weight=1.,
                 hm_weight=1.,
                 pose_weight=1.):
        super(AnchorHead, self).__init__()
        assert len(planes) in [2, 3, 4] and \
               len(planes) == len(shortcut_cfg) == len(shortcut_attention)
        self.down_ratio = 32 // 2**len(planes)

        self.planes = planes
        self.head_conv = head_conv
        self.deconv_with_bias = deconv_with_bias
        self.num_classes = num_classes
        self.num_fg = num_classes - 1

        self.use_shortcut = use_shortcut
        self.use_exp_hm = use_exp_hm
        self.use_upsample_conv = use_upsample_conv
        self.use_trident = use_trident
        self.conv_cfg = conv_cfg
        self.neg_shortcut = neg_shortcut

        self.hm_init_value = hm_init_value
        self.heights_weight = heights_weight
        self.xoff_weight = xoff_weight
        self.yoff_weight = yoff_weight
        self.hm_weight = hm_weight

        # repeat deconv n times. 32x to 4x by default.
        self.deconv_layers = nn.ModuleList([
            self._make_deconv_layer(
                inplanes[-1], 1, [planes[0]], [4], norm_cfg=norm_cfg),
            self._make_deconv_layer(
                planes[0], 1, [planes[1]], [4], norm_cfg=norm_cfg)
        ])
        for i in range(2, len(planes)):
            self.deconv_layers.append(
                self._make_deconv_layer(
                    planes[i - 1], 1, [planes[i]], [4], norm_cfg=norm_cfg))

        padding = (shortcut_kernel - 1) // 2
        self.shortcut_layers = self._make_shortcut(
            inplanes[:-1][::-1][:len(planes)],
            planes,
            shortcut_cfg,
            shortcut_attention,
            kernel_size=shortcut_kernel,
            padding=padding,
            shortcut_in_shortcut=shortcut_in_shortcut)

        # heads
        self.hm = self._make_conv_layer(
            self.num_fg, hm_head_conv_num, use_exp_conv=use_exp_hm)
        self.heights_head = self._make_conv_layer(3, wh_head_conv_num)
        self.xoffset_head = self._make_conv_layer(3, wh_head_conv_num)
        self.yoffset_head = self._make_conv_layer(3, wh_head_conv_num)
        self.pose_head = self._make_conv_layer(8, wh_head_conv_num)

        self._target_generator = CTTargetGenerator(
            self.num_fg,
            use_truncate_gaussia,
            use_tight_gauusia,
            gt_plus_dot5,
            down_ratio=self.down_ratio)
        self._loss = CTLoss(heights_weight, xoff_weight, yoff_weight,
                            hm_weight, pose_weight)

    def _make_shortcut(self,
                       inplanes,
                       planes,
                       shortcut_cfg,
                       shortcut_attention,
                       kernel_size=3,
                       padding=1,
                       shortcut_in_shortcut=False):
        assert len(inplanes) == len(planes) == len(shortcut_cfg) == len(
            shortcut_attention)

        shortcut_layers = nn.ModuleList()
        for (inp, outp, layer_num, attention) in zip(inplanes, planes,
                                                     shortcut_cfg,
                                                     shortcut_attention):
            if self.use_shortcut:
                assert layer_num > 0
                layer = ShortcutConv2d(
                    inp,
                    outp, [kernel_size] * layer_num, [padding] * layer_num,
                    use_cbam=attention,
                    shortcut_in_shortcut=shortcut_in_shortcut)
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

    def _make_deconv_layer(self,
                           inplanes,
                           num_layers,
                           num_filters,
                           num_kernels,
                           norm_cfg=None):
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
            kernel, padding, output_padding = self._get_deconv_cfg(
                num_kernels[i])
            planes = num_filters[i]
            inplanes = inplanes if i == 0 else num_filters[i - 1]

            if self.use_trident:
                mdcn = build_conv_layer(dict(type='TriConv'), inplanes, planes)
            else:
                mdcn = ModulatedDeformConvPack(
                    inplanes,
                    planes,
                    3,
                    stride=1,
                    padding=1,
                    dilation=1,
                    deformable_groups=1)
            if self.use_upsample_conv:
                up = nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
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
            head_convs.append(
                build_conv_layer(
                    dict(type='ExpConv'), out_channel, out_channel,
                    neg_x=True))
        return nn.Sequential(*head_convs)

    def fill_up_weights(self, up):
        w = up.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (
                    1 - math.fabs(j / f - c))
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

        if not self.use_exp_hm:
            self.hm[-1].bias.data.fill_(self.hm_init_value)
        else:
            self.hm[-1].conv.bias.data.fill_(self.hm_init_value)

        for m in self.heights_head.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        for m in self.xoffset_head.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        for m in self.yoffset_head.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        for m in self.pose_head.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            heatmap: tensor, (batch, cls, h, w).
            heights: tensor, (batch, 3, h, w).
            xoffset: tensor, (batch, 3, h, w).
            yoffset: tensor, (batch, 3, h, w).
            poses: tensor, (batch, 8, h, w).
            feat: tensor, (batch, c, h, w).
        """

        x = feats[-1]
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

        heatmap = self.hm(x)
        heights = self.heights_head(x)
        xoffset = self.xoffset_head(x)
        yoffset = self.yoffset_head(x)
        poses = self.pose_head(x)

        return heatmap, heights, xoffset, yoffset, poses, x

    def top_indexs(self, pred_heatmap):
        # pred_heatmap = pred_heatmap.detach()
        # wh = pred_wh.detach()
        # reg = pred_reg_offset.detach()
        pred_heatmap = pred_heatmap.sigmoid_()
        hmax = nn.functional.max_pool2d(pred_heatmap, 3, stride=1, padding=1)
        dx = (pred_heatmap - hmax) * hmax.new_tensor([1e4])
        hmax = hmax * dx.exp()

        topk = 50
        _, topk_inds = torch.topk(hmax.view(1, -1), topk)

        return hmax, topk_inds.int()

    def get_corners(self,
                    pred_heatmap,
                    pred_heights,
                    pred_xoffset,
                    pred_yoffset,
                    pred_poses,
                    img_metas,
                    cfg,
                    rescale=False):
        batch = pred_heatmap.shape[0]
        pred_heatmap = pred_heatmap.detach()
        pred_heights = pred_heights.detach()
        pred_xoffset = pred_xoffset.detach()
        pred_yoffset = pred_yoffset.detach()
        pred_poses = pred_poses.detach()

        # perform nms on heatmaps
        heat = simple_nms(
            pred_heatmap.sigmoid_())  # used maxpool to filter the max score

        topk = getattr(cfg, 'max_per_img', 100)
        # (batch, topk)
        scores, inds, clses, ys, xs = self._topk(heat, topk=topk)

        pred_heights = tranpose_and_gather_feat(pred_heights, inds)
        pred_heights = pred_heights.view(batch, topk, 3)
        pred_xoffset = tranpose_and_gather_feat(pred_xoffset, inds)
        pred_xoffset = pred_xoffset.view(batch, topk, 3, 1)
        pred_yoffset = tranpose_and_gather_feat(pred_yoffset, inds)
        pred_yoffset = pred_yoffset.view(batch, topk, 3, 1)
        pred_offset = torch.cat([pred_xoffset, pred_yoffset], dim=3)

        centers = torch.cat(
            [xs.unsqueeze(-1), ys.unsqueeze(-1)],
            dim=2).unsqueeze(dim=2) + pred_offset
        corners = centers.repeat(1, 1, 2, 1)
        corners[:, :, :3, 1] = corners[:, :, :3, 1] - pred_heights / 2
        corners[:, :, 3:, 1] = corners[:, :, 3:, 1] + pred_heights / 2
        # corners = corners[:, :, [1, 2, 5, 4, 3, 0], :]

        pose_scores, poses = F.softmax(pred_poses, 1).max(1)
        poses = poses.view(batch, -1).gather(1, inds)
        pose_scores = pose_scores.view(batch, -1).gather(1, inds)

        result_list = []
        for idx in range(batch):
            scores_per_img = scores[idx]

            scores_keep = (scores_per_img > getattr(cfg, 'score_thr',
                                                    0.01)).squeeze(-1)
            scores_per_img = scores_per_img[scores_keep]
            corners_per_img = corners[idx][scores_keep]
            labels_per_img = clses[idx][scores_keep]
            poses_per_img = poses[idx][scores_keep]
            pose_scores_per_img = pose_scores[idx][scores_keep]

            if rescale:
                scale_factor = img_metas[idx]['scale_factor'][:2]
                corners_per_img /= corners_per_img.new_tensor(
                    scale_factor).view(1, 1, 2)
            corners_per_img *= self.down_ratio

            result_list.append(
                (corners_per_img, labels_per_img, scores_per_img,
                 poses_per_img, pose_scores_per_img))

        return result_list

    def loss(self, pred_heatmap, pred_heights, pred_reg_xoffset,
             pred_reg_yoffset, pred_pose, gt_corners, gt_classes, gt_poses,
             img_metas, cfg):
        all_targets = self._target_generator(gt_corners, gt_classes, gt_poses,
                                             img_metas)
        hm_loss, heights_loss, xoff_loss, yoff_loss, pose_loss = self._loss(
            pred_heatmap, pred_heights, pred_reg_xoffset, pred_reg_yoffset,
            pred_pose, *all_targets)
        return {
            'losses/heatmap': hm_loss,
            'losses/heights': heights_loss,
            'losses/xoff': xoff_loss,
            'losses/yoff': yoff_loss,
            'losses/pose': pose_loss,
        }

    def _topk(self, scores, topk):
        batch, cat, height, width = scores.shape

        # both are (batch, 80, topk)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), topk)
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # both are (batch, topk). select topk from 80*topk
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
        topk_clses = (topk_ind / topk).int()
        topk_inds = gather_feat(topk_inds.view(batch, -1, 1),
                                topk_ind).view(batch, topk)
        topk_ys = gather_feat(topk_ys.view(batch, -1, 1),
                              topk_ind).view(batch, topk)
        topk_xs = gather_feat(topk_xs.view(batch, -1, 1),
                              topk_ind).view(batch, topk)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


class CTTargetGenerator(object):

    def __init__(self,
                 num_fg,
                 use_truncate_gaussia,
                 use_tight_gauusia,
                 gt_plus_dot5,
                 down_ratio=4,
                 max_objs=128):
        self.num_fg = num_fg
        self.use_truncate_gaussia = use_truncate_gaussia
        self.use_tight_gauusia = use_tight_gauusia
        self.gt_plus_dot5 = gt_plus_dot5
        self.down_ratio = down_ratio
        self.max_objs = max_objs

    def target_single_image(self, gt_corners, gt_classes, gt_poses,
                            feat_shape):
        """

        Args:
            gt_corners: tensor, (num_gt, 6, 2)
            gt_classes: tensor, (num_gt)
            gt_poses: tensor, (num_gt)
            feat_shape: tuple.

        Returns:
            heatmap: tensor,  (cls, h, w).
            heights: tensor, (max_obj, 3).
            xoffset: tensor, (max_obj, 3).
            yoffset: tensor, (max_obj, 3).
            pose: tensor, (max_obj).
            ind: tensor, (max_obj,).
            heatmap_mask: tensor, (cls, h, w).
            reg_mask: tensor, (max_obj,3).
            pose_mask: tensor, (max_obj,).
        """
        output_h, output_w = feat_shape
        heatmap = gt_corners.new_zeros((self.num_fg, output_h, output_w))
        heights = gt_corners.new_zeros((self.max_objs, 3))
        xoffset = gt_corners.new_zeros((self.max_objs, 3))
        yoffset = gt_corners.new_zeros((self.max_objs, 3))
        pose = gt_corners.new_zeros(self.max_objs, dtype=torch.long)
        ind = gt_corners.new_zeros((self.max_objs, ), dtype=torch.long)
        heatmap_mask = gt_corners.new_ones((self.num_fg, output_h, output_w))
        reg_mask = gt_corners.new_zeros((self.max_objs, 3))
        pose_mask = gt_corners.new_zeros((self.max_objs, ))

        num_gt = gt_corners.shape[0]
        gt_corners /= self.down_ratio
        gt_height_centers = (gt_corners[:, :3] + gt_corners[:, 3:]) / 2
        if (heights < 0).any():
            raise Exception('wrong corner order, negtive height found,' +
                            str(heights))

        xcorners = gt_corners[:, [0, 2, 3, 5], 0].clamp(0, output_w - 1)
        clamped_widths = (xcorners[:, 1] - xcorners[:, 0] + xcorners[:, 3] -
                          xcorners[:, 2]) / 2
        if (clamped_widths < 0).any():
            raise Exception('invalid clamped widths found,' +
                            str(clamped_widths))
        ycorners = gt_corners[:, [0, 2, 3, 5], 1].clamp(0, output_h - 1)
        clamped_heights = (ycorners[:, 2] - ycorners[:, 0] + ycorners[:, 3] -
                           ycorners[:, 1]) / 2
        if (clamped_heights < 0).any():
            raise Exception('invalid clamped heights found,' +
                            str(clamped_heights))
        clamp_centers = torch.stack([xcorners, ycorners], dim=2).mean(dim=1)
        centers = gt_corners[:, [0, 2, 3, 5], :].mean(dim=1)

        def inside_heatmap(point, spatial_size):
            h, w = spatial_size
            if point[0] < 0 or point[1] < 0:
                return False
            if point[0] >= w or point[1] >= h:
                return False

            return True

        valid_gt = 0
        valid_indexs = []
        is_bi_side = gt_poses.remainder(2) * (gt_poses >= 0).to(gt_poses.dtype)
        for k in range(num_gt):
            cls_id = gt_classes[k]
            # cls_id == 0 mean ignore regions
            if cls_id == 0:
                x1, y1 = gt_corners[k, 0].to(torch.int)
                x2, y2 = gt_corners[k, 5].to(torch.int)
                heatmap_mask[:, y1:y2, x1:x2] = 0
                continue


            # draw gaussian into heatmap
            h = clamped_heights[k]
            w = clamped_widths[k]
            ct_int = self._draw_heatmap(heatmap[cls_id.abs() - 1], clamp_centers[k],
                                        h, w)

            # if the class of an object is marked as negative,
            # it may not contain  enough info and we just don't regress it.
            if cls_id < 0:
                continue

            ind[valid_gt] = ct_int[1] * output_w + ct_int[0]
            if inside_heatmap(centers[k], feat_shape):
                reg_mask[valid_gt, [0, 2]] = 1
                if is_bi_side[k]:
                    reg_mask[valid_gt, 1] = 1
            else:
                if inside_heatmap(gt_corners[k, 0],
                                  feat_shape) and inside_heatmap(
                                      gt_corners[k, 3], feat_shape):
                    reg_mask[valid_gt, 0] = 1
                if inside_heatmap(gt_corners[k, 2],
                                  feat_shape) and inside_heatmap(
                                      gt_corners[k, 5], feat_shape):
                    reg_mask[valid_gt, 2] = 1
                if is_bi_side[k] and inside_heatmap(
                        gt_corners[k, 1], feat_shape) and inside_heatmap(
                            gt_corners[k, 4], feat_shape):
                    reg_mask[valid_gt, 1] = 1

            if gt_poses[k] >= 0:
                pose_mask[valid_gt] = 1
            else:
                assert reg_mask[valid_gt, 1] == 0
            valid_gt += 1
            valid_indexs.append(k)

        heights[:valid_gt] = gt_corners[valid_indexs, 3:,
                                        1] - gt_corners[valid_indexs, :3, 1]
        offset = gt_height_centers - clamp_centers.unsqueeze(dim=1)
        xoffset[:valid_gt] = offset[valid_indexs, :, 0]
        yoffset[:valid_gt] = offset[valid_indexs, :, 1]
        pose[:valid_gt] = gt_poses[valid_indexs]

        # reg_mask[:num_gt] = 1
        # reg_mask[:num_gt, 1] = is_bi_side.byte()

        return (heatmap, heights, xoffset, yoffset, pose, ind, heatmap_mask,
                reg_mask, pose_mask)

    def _draw_heatmap(self, heatmap, center, h, w):
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
            draw_truncate_gaussian(heatmap, ct_int, h_radius, w_radius)
        else:
            radius = gaussian_radius((h.ceil(), w.ceil()))
            radius = max(0, int(radius.item()))
            draw_umich_gaussian(heatmap, ct_int, radius)

        return ct_int

    def __call__(self, gt_corners, gt_classes, gt_poses, img_metas):
        """

        Args:
            gt_corners: tensor, (batch, num_gt, 6, 2).
            gt_classes: tensor, (batch, num_gt).
            gt_poses: tensor,  (batch, num_gt).
            img_metas: list(dict).

        Returns: targets = [
            heatmap: tensor,  (batch,cls, h, w),
            heights: tensor, (batch,max_obj, 3),
            xoffset: tensor, (batch,max_obj, 3),
            yoffset: tensor, (batch,max_obj, 3),
            pose: tensor, (batch,max_obj),
            ind: tensor, (batch,max_obj),
            reg_mask: tensor, (batch,max_obj,3),
            hm_mask: tensor, (batch, cls, h, w),
            pose_mask: tensor, (batch,max_obj),
            ]
        """
        feat_shape = (img_metas[0]['pad_shape'][0] // self.down_ratio,
                      img_metas[0]['pad_shape'][1] // self.down_ratio)
        targets = multi_apply(
            self.target_single_image,
            gt_corners,
            gt_classes,
            gt_poses,
            feat_shape=feat_shape)

        with torch.no_grad():
            targets = [torch.stack(t, dim=0).detach() for t in targets]

            return targets


class CTLoss(object):

    def __init__(self,
                 heights_weight=0.1,
                 xoff_weight=0.2,
                 yoff_weight=1.,
                 hm_weight=1.,
                 pose_weight=1.):
        super(CTLoss, self).__init__()
        self.heights_weight = heights_weight
        self.xoff_weight = xoff_weight
        self.yoff_weight = yoff_weight
        self.hm_weight = hm_weight
        self.pose_weight = pose_weight

    def __call__(self, pred_hm, pred_heights, pred_reg_xoffset,
                 pred_reg_yoffset, pred_pose, heatmap, heights, reg_xoffset,
                 reg_yoffset, pose, ind, hm_mask, reg_mask, pose_mask):
        """

        Args:
            pred_hm: tensor, (batch, cls, h, w).
            pred_heights: tensor, (batch, 3, h, w).
            pred_reg_xoffset: tensor, (batch, 3, h, w).
            pred_reg_yoffset: tensor, (batch, 3, h, w).
            pred_pose: tensor, (batch, 8, h, w).
            heatmap: tensor, (batch, cls, h, w).
            heights: tensor, (batch, max_obj, 3).
            reg_xoffset: tensor, (batch, max_obj, 3).
            reg_yoffset: tensor, (batch, max_obj, 3).
            pose: tensor, (batch, max_obj).
            ind: tensor, (batch, max_obj).
            hm_mask: tensor, (batch, cls, h, w).
            reg_mask: tensor, (batch, max_obj, 3).
            pose_mask: tensor, (batch, max_obj).

        Returns:

        """
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        hm_loss = ct_focal_loss(
            pred_hm, heatmap, hm_weight=hm_mask) * self.hm_weight

        # (batch, 3, h, w) => (batch, max_obj, 3)
        pred_heights = tranpose_and_gather_feat(pred_heights, ind)
        pred_reg_xoffset = tranpose_and_gather_feat(pred_reg_xoffset, ind)
        pred_reg_yoffset = tranpose_and_gather_feat(pred_reg_yoffset, ind)

        # cross_entropy only accepts (N,C,d) order
        # (batch, 8, h, w) => (batch, 8, max_obj)
        pred_pose = pred_pose.view(pred_pose.shape[0], pred_pose.shape[1], -1)
        pred_pose = pred_pose.gather(
            2,
            ind.unsqueeze(1).expand(-1, pred_pose.shape[1], -1))

        avg_factor = reg_mask.sum() + 1e-4

        heights_loss = weighted_l1(
            pred_heights, heights, reg_mask,
            avg_factor=avg_factor) * self.heights_weight
        xoff_loss = weighted_l1(
            pred_reg_xoffset, reg_xoffset, reg_mask,
            avg_factor=avg_factor) * self.xoff_weight
        yoff_loss = weighted_l1(
            pred_reg_yoffset, reg_yoffset, reg_mask,
            avg_factor=avg_factor) * self.yoff_weight

        # print('reg_mask', reg_mask[0, :10])
        # print('reg_xoffset', reg_xoffset[0, :10])
        # print('pred_reg_xoff', pred_reg_xoffset[0, :10])
        # print('gtpose ', pose[0, :10])
        # print('pred pose ', pred_pose[0, :10])
        # print('pose mask', pose_mask[0, :10])
        pose_instances = pose_mask.sum() + 1e-4
        pose_loss = cross_entropy(
            pred_pose, pose, pose_mask,
            avg_factor=pose_instances) * self.pose_weight

        return hm_loss, heights_loss, xoff_loss, yoff_loss, pose_loss
