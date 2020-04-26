import torch
import torch.nn.functional as F


def weighted_l1(pred, target, weight, avg_factor=None, reduction='sum'):
    assert reduction in ('sum', 'mean')
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    else:
        assert reduction == 'sum', "avg_factor is useless when reduction is 'mean'"
    loss = F.l1_loss(pred, target, reduction='none')
    if reduction == 'sum':
        return torch.sum(loss * weight)[None] / avg_factor
    return torch.mean(loss * weight)[None]
