import torch
import math


def giou_loss(pred,
              target,
              weight,
              avg_factor=None):
    """GIoU loss.
    Computing the GIoU loss between a set of predicted bboxes and target bboxes.
    """
    pos_mask = weight > 0
    weight = weight[pos_mask].float()
    if avg_factor is None:
        avg_factor = torch.sum(pos_mask).float().item() + 1e-6
    bboxes1 = pred[pos_mask].view(-1, 4)
    bboxes2 = target[pos_mask].view(-1, 4)

    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
    enclose_x1y1 = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    enclose_x2y2 = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)

    overlap = wh[:, 0] * wh[:, 1]
    ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    u = ap + ag - overlap
    ious = overlap / u

    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]  # i.e. C in paper
    gious = ious - (enclose_area - u) / enclose_area
    iou_distances = 1 - gious
    return torch.sum(iou_distances * weight)[None] / avg_factor


def diou_loss(pred,
              target,
              weight,
              avg_factor=None):
    pos_mask = weight > 0
    weight = weight[pos_mask].float()
    if avg_factor is None:
        avg_factor = torch.sum(pos_mask).float().item() + 1e-6
    bboxes1 = pred[pos_mask].view(-1, 4)
    bboxes2 = target[pos_mask].view(-1, 4)

    inter_x1y1 = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    inter_x2y2 = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_wh = (inter_x2y2 - inter_x1y1 + 1).clamp(min=0)

    ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    overlap = inter_wh[:, 0] * inter_wh[:, 1]
    union = ap + ag - overlap
    ious = overlap / union

    enclose_x1y1 = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    enclose_x2y2 = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    enclose_diag = (enclose_wh[:, 0] ** 2) + (enclose_wh[:, 1] ** 2)
    dious = ious - (inter_diag) / enclose_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    iou_distances = 1 - dious
    return torch.sum(iou_distances * weight)[None] / avg_factor


def ciou_loss(pred,
              target,
              weight,
              avg_factor=None):
    pos_mask = weight > 0
    weight = weight[pos_mask].float()
    if avg_factor is None:
        avg_factor = torch.sum(pos_mask).float().item() + 1e-6
    bboxes1 = pred[pos_mask].view(-1, 4)
    bboxes2 = target[pos_mask].view(-1, 4)

    inter_x1y1 = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    inter_x2y2 = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_wh = (inter_x2y2 - inter_x1y1 + 1).clamp(min=0)

    w1 = bboxes1[:, 2] - bboxes1[:, 0] + 1
    h1 = bboxes1[:, 3] - bboxes1[:, 1] + 1
    w2 = bboxes2[:, 2] - bboxes2[:, 0] + 1
    h2 = bboxes2[:, 3] - bboxes2[:, 1] + 1
    ap = w1 * h1
    ag = w2 * h2
    overlap = inter_wh[:, 0] * inter_wh[:, 1]
    union = ap + ag - overlap
    ious = overlap / union

    enclose_x1y1 = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    enclose_x2y2 = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    enclose_diag = (enclose_wh[:, 0] ** 2) + (enclose_wh[:, 1] ** 2)
    u = inter_diag / enclose_diag

    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - ious
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)

    cious = ious - (u + alpha * ar)
    cious = torch.clamp(cious, min=-1.0, max=1.0)
    iou_distances = 1 - cious
    return torch.sum(iou_distances * weight)[None] / avg_factor
