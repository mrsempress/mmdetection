import torch

from .base_assigner import BaseAssigner
from .assign_result import AssignResult
from ..geometry import bbox_overlaps


class SampledMaxIoUAssigner(BaseAssigner):

    def __init__(self,
                 min_pos_iou=.0,
                 gt_max_assign_all=True):
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all

    def assign(self,
               bboxes,
               gt_bboxes,
               sampled_wh,
               gt_labels=None):
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        num_gts, num_bboxes, num_sampled = len(gt_bboxes), len(bboxes), len(sampled_wh)

        bboxes = bboxes[:, :4]
        ctr_x = 0.5 * (bboxes[:, 2] + bboxes[:, 0])
        ctr_y = 0.5 * (bboxes[:, 3] + bboxes[:, 1])
        centers = torch.stack([
            ctr_x, ctr_y,
            ctr_x, ctr_y
        ], dim=-1).round()
        sampled_anchors = (centers[:, None, :] + sampled_wh[None, :, :]).view(-1, 4)
        # fix the location of the fake anchor and generate shape-target.
        overlaps = bbox_overlaps(sampled_anchors, gt_bboxes).view(-1, num_sampled, num_gts)
        assign_result = self.assign_wrt_overlaps(overlaps)
        return assign_result

    def assign_wrt_overlaps(self, overlaps):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(n, len(sampled), k).
        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if overlaps.numel() == 0:
            raise ValueError('No gt or proposals')

        num_gts, num_sampled, num_bboxes = overlaps.size(2), overlaps.size(1), overlaps.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps_sampled, _ = overlaps.max(dim=1)  # (num_bboxes, num_gt)
        max_overlaps, argmax_overlaps = max_overlaps_sampled.max(dim=1)  # (num_bboxes,)

        pos_inds = max_overlaps >= self.min_pos_iou
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        return AssignResult(num_gts, assigned_gt_inds, max_overlaps)
