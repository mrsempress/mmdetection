import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mmcv.cnn import normal_init
import numpy as np
from graphviz import Digraph
from collections import defaultdict

from mmdet.core import multi_apply, bbox_areas, force_fp32
from mmdet.core.anchor.guided_anchor_target import calc_region
from mmdet.core.utils.summary import add_summary, every_n_local_step, add_histogram_summary
from mmdet.models.losses import ct_focal_loss, giou_loss
from mmdet.models.utils import (
    bias_init_with_prob, ConvModule, simple_nms, OPS)
from .anchor_head import AnchorHead
from ..registry import HEADS


@HEADS.register_module
class TTFXHead(AnchorHead):

    def __init__(self,
                 genotype=None,
                 search_k=4,
                 search_op_gumbel_softmax=False,
                 search_edge_gumbel_softmax=False,
                 gumbel_no_affine=False,
                 tau=10.,
                 multiply=None,
                 has_beta=True,
                 inplanes=(64, 128, 256, 512),
                 planes=(64, 128, 256, 512),
                 head_conv=256,
                 wh_conv=64,
                 hm_head_conv_num=2,
                 wh_head_conv_num=2,
                 num_classes=81,
                 ops_list=[],
                 wh_offset_base=16.,
                 wh_gaussian=True,
                 alpha=0.54,
                 beta=0.54,
                 hm_weight=1.,
                 wh_weight=5.,
                 max_objs=128,
                 **kwargs):
        super(AnchorHead, self).__init__()
        assert len(planes) == 4

        self.search_k = search_k
        if search_op_gumbel_softmax == search_edge_gumbel_softmax == True and has_beta:
            assert 'none' not in ops_list
        self.search_op_gumbel_softmax = search_op_gumbel_softmax
        self.search_edge_gumbel_softmax = search_edge_gumbel_softmax
        if multiply == None:
            multiply = float(len(ops_list)) if has_beta else 1.
        self.multiply = multiply
        self.has_beta = has_beta
        self.inplanes = inplanes
        self.planes = planes
        self.head_conv = head_conv
        self.num_classes = num_classes
        self.ops_list = ops_list
        self.wh_offset_base = wh_offset_base
        self.wh_gaussian = wh_gaussian
        self.alpha = alpha
        self.beta = beta
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.max_objs = max_objs
        self.fp16_enabled = False

        self.down_ratio = 4
        self.num_fg = num_classes - 1
        self.wh_planes = 4
        self.base_loc = None

        if isinstance(genotype, str):
            assert os.path.exists(genotype) and os.path.basename(genotype) == 'search.pkl'
            genotype = pickle.load(open(genotype, 'rb'))

        self.auto_head = SearchSpace(inplanes, planes, search_k, search_op_gumbel_softmax,
                                     search_edge_gumbel_softmax, tau, ops_list, multiply, has_beta,
                                     gumbel_no_affine, genotype=genotype, **kwargs)

        # heads
        self.wh = self.build_head(self.wh_planes, wh_head_conv_num, wh_conv)
        self.hm = self.build_head(self.num_fg, hm_head_conv_num)

    def build_head(self, out_channel, conv_num=1, head_conv_plane=None):
        head_convs = []
        head_conv_plane = self.head_conv if not head_conv_plane else head_conv_plane
        for i in range(conv_num):
            inp = self.planes[0] if i == 0 else head_conv_plane
            head_convs.append(ConvModule(inp, head_conv_plane, 3, padding=1))

        inp = self.planes[0] if conv_num <= 0 else head_conv_plane
        head_convs.append(nn.Conv2d(inp, out_channel, 1))
        return nn.Sequential(*head_convs)

    def rebuild(self, g):
        raise NotImplementedError
        # del self.auto_head
        # self.auto_head = SearchSpace(self.inplanes, self.planes, self.search_k,
        #                              self.search_op_gumbel_softmax,
        #                              self.search_edge_gumbel_softmax, self.ops_list,
        #                              self.multiply, g).cuda()

    def reset_tau(self, tau):
        return self.auto_head.reset_tau(tau)

    def reset_do_search(self, do_search):
        return self.auto_head.reset_do_search(do_search)

    def reset_gumbel_buffer(self):
        return self.auto_head.reset_gumbel_buffer()

    def init_weights(self):
        self.auto_head.init_weight()

        for _, m in self.hm.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.hm[-1], std=0.01, bias=bias_cls)

        for _, m in self.wh.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def arch_parameters(self):
        return self.auto_head.arch_parameters

    def genotype(self):
        return self.auto_head.genotype()

    def plot(self, g, filename):
        return self.auto_head.plot(g, filename)

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        x = self.auto_head(feats)
        hm = self.hm(x)
        wh = F.relu(self.wh(x)) * self.wh_offset_base

        return hm, wh

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh'))
    def get_bboxes(self,
                   pred_heatmap,
                   pred_wh,
                   img_metas,
                   cfg,
                   rescale=False):
        batch, cat, height, width = pred_heatmap.size()
        pred_heatmap = pred_heatmap.detach().sigmoid_()
        wh = pred_wh.detach()

        # perform nms on heatmaps
        heat = simple_nms(pred_heatmap)  # used maxpool to filter the max score

        topk = getattr(cfg, 'max_per_img', 100)
        # (batch, topk)
        scores, inds, clses, ys, xs = self._topk(heat, topk=topk)
        xs = xs.view(batch, topk, 1) * self.down_ratio
        ys = ys.view(batch, topk, 1) * self.down_ratio

        wh = wh.permute(0, 2, 3, 1).contiguous()
        wh = wh.view(wh.size(0), -1, wh.size(3))
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), wh.size(2))
        wh = wh.gather(1, inds)

        wh = wh.view(batch, topk, 4)
        clses = clses.view(batch, topk, 1).float()
        scores = scores.view(batch, topk, 1)

        bboxes = torch.cat([xs - wh[..., [0]], ys - wh[..., [1]],
                            xs + wh[..., [2]], ys + wh[..., [3]]], dim=2)

        result_list = []
        score_thr = getattr(cfg, 'score_thr', 0.01)
        for idx in range(bboxes.shape[0]):
            scores_per_img = scores[idx]
            scores_keep = (scores_per_img > score_thr).squeeze(-1)

            scores_per_img = scores_per_img[scores_keep]
            bboxes_per_img = bboxes[idx][scores_keep]
            labels_per_img = clses[idx][scores_keep]

            if rescale:
                scale_factor = img_metas[idx]['scale_factor']
                bboxes_per_img /= bboxes_per_img.new_tensor(scale_factor)

            bboxes_per_img = torch.cat([bboxes_per_img, scores_per_img], dim=1)
            labels_per_img = labels_per_img.squeeze(-1)
            result_list.append((bboxes_per_img, labels_per_img))

        return result_list

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh'))
    def loss(self,
             pred_heatmap,
             pred_wh,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        all_targets = self.target_generator(gt_bboxes, gt_labels, img_metas)
        hm_loss, wh_loss = self.loss_calc(pred_heatmap, pred_wh, *all_targets)
        return {'losses/ttfnet_loss_heatmap': hm_loss, 'losses/ttfnet_loss_wh': wh_loss}

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

        boxes_areas_log = bbox_areas(gt_boxes).log()
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))

        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        feat_gt_boxes = gt_boxes / self.down_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0, max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0, max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
        if self.alpha != self.beta:
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

            if self.alpha != self.beta:
                fake_heatmap = fake_heatmap.zero_()
                self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                            h_radiuses_beta[k].item(), w_radiuses_beta[k].item())
            if self.wh_gaussian:
                box_target_inds = fake_heatmap > 0
            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[k], ctr_y1s[k], ctr_x2s[k], ctr_y2s[k]
                box_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1

            box_target[:, box_target_inds] = gt_boxes[k][:, None]

            local_heatmap = fake_heatmap[box_target_inds]
            ct_div = local_heatmap.sum()
            local_heatmap *= boxes_area_topk_log[k]

            cls_id = 0
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
        wh_loss = giou_loss(pred_boxes, boxes, mask, avg_factor=avg_factor) * self.wh_weight

        return hm_loss, wh_loss


class SearchSpace(nn.Module):

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes=(64, 128, 256, 512),
                 search_k=4,
                 search_op_gumbel_softmax=False,
                 search_edge_gumbel_softmax=False,
                 tau=10.,
                 ops_list=[],
                 multiply=10.,
                 has_beta=True,
                 gumbel_no_affine=False,
                 genotype=None,
                 force_n=None):
        super(SearchSpace, self).__init__()
        assert len(inplanes) == len(planes) == 4
        self.pre_convs = nn.ModuleList([nn.Conv2d(inplane, plane, 3, padding=1)
                                        for (inplane, plane) in zip(inplanes, planes)])
        self.last_bn = nn.BatchNorm2d(planes[0], affine=True)
        self.ops_list = ops_list
        self.is_rebuilded = False
        self.search_op_gumbel_softmax = search_op_gumbel_softmax
        self.search_edge_gumbel_softmax = search_edge_gumbel_softmax
        self.force_n = force_n
        self.tau = tau
        assert self.tau == 10.
        self.multiply = multiply
        self.has_beta = has_beta
        self.do_search = False
        # I - O - O - O
        #   \/  \/  /
        # I - O - O
        #   \/  /
        # I - O
        #   /
        # I
        # level: 3
        # edge: 8, 5, 2
        # 3 below stands for the max edges num for each node
        level = len(planes) - 1
        self.ops = nn.ModuleList([nn.ModuleList() for _ in range(level)])

        self.node_num_per_level = (3, 2, 1)
        self.edge_num_per_level = (8, 5, 2)
        self.edges_idx_for_node = ((0, 1), (2, 3, 4), (5, 6, 7), (8, 9), (10, 11, 12), (13, 14))

        self.node_idx_for_edge = (0, 1, 0, 1, 2, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8)  # 0 ~ 3 => inputs
        self.in_stride_for_edge = (0, 1, 0, 1, 2, 1, 2, 3, 0, 1, 0, 1, 2, 0, 1)
        self.out_stride_for_edge = (0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 1, 0, 0)

        self.edge_num_for_node = [len(edges_idx) for edges_idx in self.edges_idx_for_node]
        self.total_node_num = sum(self.node_num_per_level)  # 6, node0 ~ node3 are excluded
        assert len(self.edges_idx_for_node) == self.total_node_num and len(
            self.node_idx_for_edge) == len(self.in_stride_for_edge) == len(self.out_stride_for_edge)

        # auto-set vars
        edge_up, edge_down = [], []
        for i, j in zip(self.in_stride_for_edge, self.out_stride_for_edge):
            edge_up.append(True if i > j else False)
            edge_down.append(True if i < j else False)

        out_level_for_edge = [0]
        pre_stride = self.out_stride_for_edge[0]
        for out_stride in self.out_stride_for_edge[1:]:
            if out_stride < pre_stride:
                out_level_for_edge.append(out_level_for_edge[-1] + 1)
            else:
                out_level_for_edge.append(out_level_for_edge[-1])
            pre_stride = out_stride

        if genotype is None:
            edge_idx = 0
            for i in range(level):
                for j in range(self.edge_num_per_level[i]):
                    self.ops[i].append(
                        MixedOp(planes[self.in_stride_for_edge[edge_idx]],
                                planes[self.out_stride_for_edge[edge_idx]],
                                ops_list,
                                gumbel_no_affine,
                                k=search_k,
                                use_gumbel=search_op_gumbel_softmax,
                                is_up=edge_up[edge_idx],
                                is_down=edge_down[edge_idx]))
                    edge_idx += 1
        else:
            assert len(genotype) % 2 == 0 and len(genotype) // 2 == self.total_node_num
            self.is_rebuilded = True
            for node_idx in range(self.total_node_num):
                op_name_1, in_node_idx_1 = genotype[2 * node_idx]
                op_name_2, in_node_idx_2 = genotype[2 * node_idx + 1]
                in_edges = self.edges_idx_for_node[node_idx]
                in_nodes = [self.node_idx_for_edge[edge_j] for edge_j in in_edges]

                for in_node_idx, in_edge_idx in zip(in_nodes, in_edges):
                    op_name = 'none'
                    if in_node_idx == in_node_idx_1:
                        op_name = op_name_1
                    elif in_node_idx == in_node_idx_2:
                        op_name = op_name_2
                    self.ops[out_level_for_edge[in_edge_idx]].append(
                        SelectedOp(planes[self.in_stride_for_edge[in_edge_idx]],
                                   planes[self.out_stride_for_edge[in_edge_idx]],
                                   op_name,
                                   is_up=edge_up[in_edge_idx],
                                   is_down=edge_down[in_edge_idx]))

        self.initialize_arch_params(requires_grad=genotype is None)
        self.init_weight()

        self.alpha_buffer = defaultdict(dict)
        self.beta_buffer = defaultdict(dict)

    def initialize_arch_params(self, requires_grad=True):
        num_edges = sum(1 for i in range(len(self.ops)) for _ in self.ops[i])
        num_ops = len(self.ops_list)

        self.register_buffer('search_alphas', Variable(1e-3 * torch.randn(
            num_edges, num_ops).cuda(), requires_grad=requires_grad))
        self.arch_parameters = [self.search_alphas]
        self.register_buffer('search_betas', Variable(1e-3 * torch.randn(
            num_edges).cuda(), requires_grad=requires_grad and self.has_beta))
        if self.has_beta:
            self.arch_parameters.append(self.search_betas)

    def init_weight(self):
        for name, m in self.ops.named_modules():
            if isinstance(m, nn.Conv2d) and not name.endswith('conv_offset'):
                normal_init(m, std=0.01)
            if isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def reset_tau(self, tau):
        self.tau = tau

    def reset_do_search(self, do_search):
        self.do_search = do_search

    def reset_gumbel_buffer(self):
        self.alpha_buffer = defaultdict(dict)
        self.beta_buffer = defaultdict(dict)

    def forward(self, feats):
        # print(F.softmax(self.search_alphas, dim=-1), F.softmax(self.search_betas))
        pre_feats = []
        for i, op in enumerate(self.pre_convs):
            pre_feats.append(op(feats[i]))
        # if self.do_search and self.search_op_gumbel_softmax:
        #     samples_weight = self.gumbel_softmax(self.search_alphas, self.tau)
        edge_idx, beta_start_idx = 0, 0
        alpha_summary, beta_summary = {}, {}
        alphas_softmax = F.softmax(self.search_alphas * self.multiply, dim=-1)
        for level_i, ops_level_i in enumerate(self.ops):
            post_feats = [[] for _ in range(self.node_num_per_level[level_i])]
            for j, op in enumerate(ops_level_i):
                ops_weight = alphas_softmax[edge_idx]
                for idx, w in enumerate(ops_weight):
                    alpha_summary['edge_{}_{}'.format(edge_idx, idx)] = w.item()

                if self.do_search and self.search_op_gumbel_softmax:
                    if j in self.alpha_buffer[level_i]:
                        ops_weight = self.alpha_buffer[level_i][j].to(ops_weight.device)
                    else:
                        ops_weight = self.gumbel_softmax(
                            self.search_alphas[edge_idx] * self.multiply, self.tau, topk=1)
                        self.alpha_buffer[level_i][j] = torch.tensor(ops_weight)

                post_feats[self.out_stride_for_edge[edge_idx]].append(
                    op(pre_feats[self.in_stride_for_edge[edge_idx]], ops_weight,
                       'edge_{}_'.format(edge_idx)))
                edge_idx += 1

            for stride_j, post_feats_n in enumerate(post_feats):
                beta_end_idx = beta_start_idx + len(post_feats_n)
                if not self.is_rebuilded:
                    edges_weight = F.softmax(self.search_betas[beta_start_idx:beta_end_idx])
                    for idx, w in enumerate(edges_weight):
                        beta_summary['edge_{}'.format(beta_start_idx + idx)] = w.item()

                    # if every_n_local_step(200):
                    #     for k, feat in enumerate(post_feats_n):
                    #         add_histogram_summary('edge_{}'.format(beta_start_idx + k),
                    #                               feat.detach().cpu())
                    if self.do_search and self.search_edge_gumbel_softmax:
                        if stride_j in self.beta_buffer[level_i]:
                            edges_weight = self.beta_buffer[level_i][stride_j].to(
                                edges_weight.device)
                        else:
                            if self.has_beta:
                                edges_weight = self.gumbel_softmax(
                                    self.search_betas[beta_start_idx:beta_end_idx], self.tau,
                                    topk=2)
                            else:
                                edges_weight = self.gumbel_softmax(
                                    self.search_alphas.max(-1)[0][beta_start_idx:beta_end_idx] * \
                                        self.multiply, self.tau, topk=2)
                            self.beta_buffer[level_i][stride_j] = torch.tensor(edges_weight)
                else:
                    edges_weight = [feats[0].new_ones((1,)) for _ in post_feats_n]
                post_feats[stride_j] = sum(post_feat * w for i, (post_feat, w) in enumerate(
                    zip(post_feats_n, edges_weight)))
                beta_start_idx = beta_end_idx

            pre_feats = post_feats

        if not self.is_rebuilded:
            add_summary('alphas', **alpha_summary)
            add_summary('betas', **beta_summary)
            raw_summary = {'tau': self.tau,
                           'alpha': torch.mean(torch.abs(self.search_alphas))}
            if self.has_beta:
                raw_summary['beta'] = torch.mean(torch.abs(self.search_betas))
            add_summary('raw', **raw_summary)

        return self.last_bn(post_feats[0])

    def gumbel_softmax(self, logits, temperature, topk=1):
        """
        input: (..., choice_num)
        return: (..., choice_num), an one-zero vector
        """
        if logits.shape[-1] == topk:
            return torch.ones_like(logits) / topk
        shape = logits.size()
        if self.training:
            logits = logits.softmax(-1)
            if logits.dim() == 1:
                logits_topk, logits_inds = logits.topk(topk + 1, dim=-1)
                # print(logits, logits_topk, logits_topk[..., -2], logits_topk[..., -1])
                if self.force_n and logits_topk[..., -2] > self.force_n * logits_topk[..., -1]                                              :
                    y_hard = torch.zeros_like(logits).view(-1, shape[-1])
                    y_hard.scatter_(1, logits_inds[..., :-1].view(-1, topk), 1)
                    y_hard = y_hard.view(*shape)
                    # print(y_hard / topk)
                    return y_hard / topk
            empty_tensor = logits.new_zeros(logits.size())
            U = nn.init.uniform_(empty_tensor)
            gumbel_sample = -Variable(torch.log(-torch.log(U + 1e-20) + 1e-20))
            y = F.softmax((logits.log() + gumbel_sample) / temperature, dim=-1)
            # print(logits, gumbel_sample)
        else:
            y = logits
        _, inds = y.topk(topk, dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, inds.view(-1, topk), 1)
        y_hard = y_hard.view(*shape)
        if self.training:
            return ((y_hard - y).detach() + y) / topk
        return y_hard / topk

    def genotype(self):

        def _parse(alphas, betas):
            gene = []
            _beta_start_idx = 0
            for edge_num in self.edge_num_for_node:
                _beta_end_idx = _beta_start_idx + edge_num
                W = alphas[_beta_start_idx:_beta_end_idx].copy()

                if (self.has_beta and not self.search_edge_gumbel_softmax and
                    not self.search_op_gumbel_softmax) or not self.has_beta:
                    if self.has_beta:
                        W2 = betas[_beta_start_idx:_beta_end_idx].copy()
                        for j in range(edge_num):
                            W[j, :] = W[j, :] * W2[j]
                    # based on the max score of non-none op
                    edges = sorted(range(edge_num), key=lambda x: -max(
                        W[x][k] for k in range(len(W[x])) if
                        ('none' not in self.ops_list or k != self.ops_list.index('none'))))[:2]
                    for j in edges:
                        k_best_op = 'none'
                        for k in range(len(W[j])):
                            if 'none' not in self.ops_list or k != self.ops_list.index('none'):
                                if k_best_op is 'none' or W[j][k] > W[j][k_best_op]:
                                    k_best_op = k
                        gene.append(
                            (self.ops_list[k_best_op],
                             self.node_idx_for_edge[_beta_start_idx + j]))
                else:
                    W2 = betas[_beta_start_idx:_beta_end_idx].copy()
                    edges = sorted(range(edge_num), key=lambda x: -W2[x])[:2]
                    for j in edges:
                        k_best_op = 'none'
                        for k in range(len(W[j])):
                            # we allow the 'none' to be the best
                            if k_best_op is 'none' or W[j][k] > W[j][k_best_op]:
                                k_best_op = k
                        gene.append(
                            (self.ops_list[k_best_op],
                             self.node_idx_for_edge[_beta_start_idx + j]))

                _beta_start_idx = _beta_end_idx
            return gene

        search_betas = None
        if self.has_beta:
            beta_start_idx = 0
            search_betas = self.search_betas.new_zeros((0,))
            for edge_num in self.edge_num_for_node:
                beta_end_idx = beta_start_idx + edge_num
                search_betas = torch.cat([search_betas, F.softmax(
                    self.search_betas[beta_start_idx:beta_end_idx], dim=-1)])
                beta_start_idx = beta_end_idx
            search_betas = search_betas.data.cpu().numpy()

        gene = _parse(F.softmax(self.search_alphas * self.multiply, dim=-1).data.cpu().numpy(),
                      search_betas)

        return gene

    def plot(self, genotype, filename='search_graph'):
        g = Digraph(
            format='pdf',
            edge_attr=dict(fontsize='20', fontname="times"),
            node_attr=dict(style='filled', shape='rect', align='center', fontsize='20',
                           height='0.5', width='0.5', penwidth='2', fontname="times"),
            engine='dot')
        g.body.extend(['rankdir=LR'])

        g.node("c_s4", fillcolor='darkseagreen2')
        g.node("c_s8", fillcolor='darkseagreen2')
        g.node("c_s16", fillcolor='darkseagreen2')
        g.node("c_s32", fillcolor='darkseagreen2')

        assert len(genotype) % 2 == 0 and \
               self.total_node_num == len(genotype) // 2  # 2 inputs => op => 1 output

        for i in range(self.total_node_num):
            g.node(str(i), fillcolor='lightblue')

        for i in range(self.total_node_num):
            for k in [2 * i, 2 * i + 1]:
                op, j = genotype[k]
                if j == 0:
                    u = "c_s4"
                elif j == 1:
                    u = "c_s8"
                elif j == 2:
                    u = "c_s16"
                elif j == 3:
                    u = "c_s32"
                else:
                    u = str(j - 4)
                v = str(i)
                g.edge(u, v, label=op, fillcolor="gray")

        g.render(filename, view=False)


class MixedOp(nn.Module):

    def __init__(self, inc, outc, ops_list, gumbel_no_affine, k=4, use_gumbel=False, is_up=False,
                 is_down=False):
        super(MixedOp, self).__init__()
        assert not (is_up and is_down)
        self._ops = nn.ModuleList()
        self.k = k  # partial channels
        self.use_gumbel = use_gumbel
        self.is_up = is_up
        self.is_down = is_down

        self.inc = inc
        self.outc = outc

        for op_name in ops_list:
            op = OPS[op_name](inc // k, outc // k, stride=2 if is_down else 1,
                              upsample=is_up, affine=use_gumbel and not gumbel_no_affine)
            self._ops.append(op)

        if not self.use_gumbel:
            self.bn = nn.BatchNorm2d(outc, affine=False)
        elif gumbel_no_affine:
            self.bn = nn.BatchNorm2d(outc, affine=True)

        if k > 1 and is_up:
            self.upsample = nn.Sequential(
                # nn.ReLU(inplace=False),
                nn.Conv2d(inc // k * (k - 1), outc // k * (k - 1), 1),
                nn.UpsamplingBilinear2d(scale_factor=2),
                # nn.BatchNorm2d(outc // k * (k - 1), affine=use_gumbel)
            )
        elif k > 1 and is_down:
            self.downsample = nn.Sequential(
                # nn.ReLU(inplace=False),
                nn.Conv2d(inc // k * (k - 1), outc // k * (k - 1), 3, stride=2, padding=1),
                # nn.BatchNorm2d(outc // k * (k - 1), affine=use_gumbel)
            )

    def forward(self, x, weights, summary_prefix=''):
        channel = x.shape[1]
        x1 = x[:, :channel // self.k, :, :]
        feats = []
        for i, (op, w) in enumerate(zip(self._ops, weights)):
            if w > 1e-4:
                feat = op(x1)
                feats.append(feat)
            else:
                feats.append(0.)

            # if not self.is_rebuilded:
            #     if every_n_local_step(200):
            #         add_histogram_summary(summary_prefix + 'op_{}'.format(i), feat.detach().cpu())

        y = sum(feat * w if w > 0.01 else w for feat, w in zip(feats, weights))
        if self.k > 1:
            x2 = x[:, channel // self.k:, :, :]
            if self.is_up:
                x2 = self.upsample(x2)
            elif self.is_down:
                x2 = self.downsample(x2)
            y = torch.cat([y, x2], dim=1)
            y = self.channel_shuffle(y, self.k)

        if y.dim() == 4:
            if hasattr(self, 'bn'):
                y = self.bn(y)
        return y

    def channel_shuffle(self, x, groups):
        if groups == 1:
            return x
        N, C, H, W = x.data.size()
        C_per_group = C // groups
        x = x.view(N, groups, C_per_group, H, W)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(N, -1, H, W)
        return x


class SelectedOp(nn.Module):

    def __init__(self, inc, outc, op_name, is_up=False, is_down=False):
        super(SelectedOp, self).__init__()
        assert not (is_up and is_down)
        self._op = OPS[op_name](inc, outc, stride=2 if is_down else 1,
                                upsample=is_up, affine=True)

    def forward(self, x, weights=None, summary_prefix=''):
        y = self._op(x)
        return y
