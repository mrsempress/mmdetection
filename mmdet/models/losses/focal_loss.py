import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss


# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          clamp=None,
                          use_sigmoid=True,
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid() if use_sigmoid else pred
    if clamp:
        pred_sigmoid = torch.clamp(pred_sigmoid, min=clamp[0], max=clamp[1])
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)  # pt is (1-pt) in origin paper
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)  # alpha_t * (1 - pt) ** gamma
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def ct_focal_loss(pred, gt, gamma=2.0, weight=1.0, beta=4, hm_weight=None):
    """
    Focal loss used in CenterNet. Note that the values in gt (label) are in [0, 1] since
    gaussian is used to reduce the punishment and we treat [0, 1) as neg example.

    Args:
        pred: tensor, any shape.
        gt: tensor, same as pred.
        gamma: gamma in focal loss.
        weight: int or tensor, same as pred.

    Returns:

    """
    if hm_weight is None:
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, beta)  # reduce punishment
        pos_loss = -torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * neg_inds * weight

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return neg_loss
        return (pos_loss + neg_loss) / num_pos
    else:
        pos_inds = gt.gt(0.9).float()
        neg_inds = gt.le(0.9).float()

        pos_weights = torch.pow(gt, 4) * pos_inds  # reduce punishment
        neg_weights = torch.pow(1 - gt, beta) * neg_inds
        pos_loss = -torch.log(pred) * torch.pow(1 - pred, gamma) * pos_weights * hm_weight
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights

        ct_sum = gt.eq(1).float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if ct_sum < 1e-4:
            return neg_loss
        return (pos_loss + neg_loss) / ct_sum


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred, target, gamma, alpha)
    # TODO: find a proper way to handle the shape of weight
    if weight is not None:
        weight = weight.view(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
