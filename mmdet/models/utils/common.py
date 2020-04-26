from collections import OrderedDict
import math
import numpy as np
import functools
import torch
import torch.nn as nn
from mmdet.models.losses import py_sigmoid_focal_loss
from mmdet.models.utils import build_norm_layer


def yolov3_conv2d(inplanes,
                  planes,
                  kernel,
                  padding,
                  stride,
                  norm_cfg=dict(type='BN')):
    """A common conv-bn-leakyrelu cell"""
    cell = OrderedDict()
    cell['conv'] = nn.Conv2d(inplanes, planes, kernel_size=kernel,
                             stride=stride, padding=padding, bias=False)
    if norm_cfg:
        norm_name, norm = build_norm_layer(norm_cfg, planes)
        cell[norm_name] = norm

    cell['leakyrelu'] = nn.LeakyReLU(0.1)
    cell = nn.Sequential(cell)
    return cell


class ShareConv(nn.Module):

    def __init__(self, inchannel, outchannel):
        super(ShareConv, self).__init__()
        self.weight = nn.Parameter(torch.randn(inchannel, outchannel, 3, 3))
        nn.init.normal_(self.weight, 0, 0.01)

    def forward(self, x):
        x1 = nn.functional.conv2d(x, self.weight, bias=None, stride=1, padding=1, dilation=1,
                                  groups=1)
        x2 = nn.functional.conv2d(x, self.weight, bias=None, stride=1, padding=2, dilation=2,
                                  groups=1)
        return x1, x2


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return torch.min(torch.min(r1, r2), r3)


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = torch.arange(0, size, 1, torch.float)
    y = x[:, None]
    x0 = y0 = size // 2
    g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = torch.max(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma_x=diameter / 6, sigma_y=diameter / 6)
    gaussian = heatmap.new_tensor(gaussian)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_truncate_gaussian(heatmap, center, h_radius, w_radius, k=1):
    h, w = 2 * h_radius + 1, 2 * w_radius + 1
    sigma_x = w / 6
    sigma_y = h / 6
    gaussian = gaussian2D((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
    gaussian = heatmap.new_tensor(gaussian)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, w_radius), min(width - x, w_radius + 1)
    top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius - left:w_radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def extra_path(start=None, args_num=2, extra_num=None):
    """
    A decorator which create a clean path for feature or other params to pass.
    This decorator will check the input, compare len(args) and args_num to figure out
    if there are any extra params. If True, the decorator will select extra params and
    concat them to the return results.

    Args:
        start: the possible start index of extra params.
        args_num: same as len(args) of func.
        extra_num: used in another situation.

    Returns:

    """
    def deco(func):
        @functools.wraps(func)
        def wraper(*args, **kwargs):
            if start:
                extra_args_num = len(args) - args_num
                extra_args = args[start:start + extra_args_num]
                args = args[:start] + args[start + extra_args_num:]
            elif extra_num:
                # args: tuple(tuple(tensor), xx).
                assert len(args[1]) == 2, NotImplementedError
                extra_args = [args[1][1]]
                args = (args[0], args[1][0])
            else:
                raise NotImplementedError

            out = func(*args, **kwargs)

            if not isinstance(out, (tuple, list)):
                out = [out]

            if len(extra_args) > 1:
                extra_args = (extra_args,)

            return tuple(out) + tuple(extra_args)
        return wraper

    return deco


def extra_mask_loss(mask_idx, gt_boxes_idx):
    """

    Args:
        mask_idx: idx of args with the mask info.
        gt_boxes_idx: idx of args with gt_boxes info. Note that we do not need to take
            mask into consideration.

    Returns:

    """
    def deco(func):
        @functools.wraps(func)
        def wraper(*args, **kwargs):
            mask = args[mask_idx]
            args = args[:mask_idx] + args[mask_idx + 1:]
            gt_boxes = args[gt_boxes_idx]
            out = func(*args, **kwargs)
            mask, mask_corner, feat_stride = mask

            # loss for heatmap
            gt_heatmap = heatmap_target(mask.shape[2:], gt_boxes, feat_stride)
            weight = (mask >= 0).to(mask.dtype)
            ht_loss = py_sigmoid_focal_loss(mask, gt_heatmap, weight=weight, use_sigmoid=False)

            # TODO loss for corner
            # gt_corner = corner_target(mask.shape[2:], gt_boxes, feat_stride)
            # weight = mask_corner.new_ones(mask_corner.size())
            # corner_loss = giou_loss(mask_corner, gt_corner, weight)

            mask_loss = {'losses/mask_heatmap': ht_loss}
            out.update(mask_loss)
            return out

        return wraper

    return deco


def heatmap_target(feat_hw, gt_boxes, feat_stride):
    if not isinstance(gt_boxes, (tuple, list)):
        gt_boxes = [gt_boxes]

    with torch.no_grad():
        heatmaps = []
        for gt_boxes_img in gt_boxes:
            heatmap = gt_boxes_img.new_zeros(feat_hw)
            for gt_box in gt_boxes_img:
                x1, y1, x2, y2 = (gt_box / feat_stride).int()
                x1, y1 = x1.clamp(min=0), y1.clamp(min=0)
                x2, y2 = x2.clamp(max=feat_hw[1]), y2.clamp(max=feat_hw[0])
                # since we don't have the segment info, so we ignore the box area.
                heatmap[y1:y2, x1:x2] = -1
                # but we can suppose the center is pos at least.
                x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
                heatmap[x_center, y_center] = 1
            heatmaps.append(heatmap)
        return torch.stack(heatmaps, dim=0).unsqueeze(1)


def corner_target(feat_hw, gt_boxes, feat_stride):
    if not isinstance(gt_boxes, (tuple, list)):
        gt_boxes = [gt_boxes]

    with torch.no_grad():
        corners = []
        for gt_boxes_img in gt_boxes:
            x1, y1, x2, y2 = (gt_boxes_img / feat_stride).split(1, dim=1)
            x1, y1 = x1.min().clamp(min=0), y1.min().clamp(min=0)
            x2, y2 = x2.max().clamp(max=feat_hw[1]), y2.max().clamp(max=feat_hw[0])
            corner = torch.stack([x1, y1, x2, y2], dim=0)
            corners.append(corner)

        return torch.stack(corners, dim=0)


def left_aggregate(heat):
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])  # (batch*c*h, w)
    heat = heat.transpose(1, 0).contiguous()  # (w, batch*c*h)
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def right_aggregate(heat):
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])  # (batch*c*h, w)
    heat = heat.transpose(1, 0).contiguous()  # (w, batch*c*h)
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i + 1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def top_aggregate(heat):
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])  # (batch*c*w, h)
    heat = heat.transpose(1, 0).contiguous()  # (h, batch*c*w)
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def bottom_aggregate(heat):
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])  # (batch*c*w, h)
    heat = heat.transpose(1, 0).contiguous()  # (h, batch*c*w)
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i + 1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def h_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * (left_aggregate(heat) + right_aggregate(heat)) + heat


def v_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * (top_aggregate(heat) + bottom_aggregate(heat)) + heat


def simple_nms(heat, kernel=3, out_heat=None):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    out_heat = heat if out_heat is None else out_heat
    return out_heat * keep
