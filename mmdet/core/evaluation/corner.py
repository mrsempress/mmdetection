import numpy as np

from mmdet.ops import nms
from mmdet.core.evaluation.mean_ap import tpfpfn_default as tpfp_func


def corners2bboxes(corners):
    bboxes = []
    for i in range(len(corners)):
        corners_per_object = corners[i]
        x1 = corners_per_object[[0, 2, 3, 5], 0].min()
        x2 = corners_per_object[[0, 2, 3, 5], 0].max()
        y1 = corners_per_object[[0, 2, 3, 5], 1].min()
        y2 = corners_per_object[[0, 2, 3, 5], 1].max()
        bboxes.append([x1, y1, x2, y2])

    return np.array(bboxes, dtype=np.float32)


def filter_results(results, score_thr=0.2, iou_thr=0.65):
    det_corners = []
    det_bboxes = []
    det_poses = []
    for r in results:
        if len(r) == 5:
            corners, _, scores, poses, pose_scores = r
        else:
            corners, _, scores, poses = r
        bboxes = corners2bboxes(corners)
        bboxes = np.hstack([bboxes, scores[:, np.newaxis]])
        bboxes, nms_idx = nms(bboxes, iou_thr)

        selected = bboxes[:, -1] > score_thr
        det_bboxes.append(bboxes[selected])
        det_corners.append(corners[nms_idx][selected])
        det_poses.append(poses[nms_idx][selected])

    return det_corners, det_bboxes, det_poses


def corners_nms(det_corners, det_bboxes, iou_thr=0.5):
    det_corners_nms = []
    det_bboxes_nms = []
    for corners, bboxes in zip(det_corners, det_bboxes):
        bboxes, nms_idx = nms(bboxes, iou_thr)
        det_corners_nms.append(corners[nms_idx])
        det_bboxes_nms.append(bboxes)

    return det_corners_nms, det_bboxes_nms


def pose2index(pose):
    if pose % 2 == 1:
        return range(6)
    else:
        return [0, 2, 3, 5]


def find_vehicle(labels):
    vehicle_idx = []
    for i in range(len(labels)):
        if labels[i] in [1, 2, 3, 6]:
            vehicle_idx.append(i)

    return vehicle_idx


def eval_corners(raw_results,
                 gt_corners,
                 gt_poses,
                 gt_labels,
                 gt_ignores,
                 gt_bboxes=None,
                 with_nms=True,
                 display=True):
    """
    Args:
        gt_corners: list(np.array), img,(num_gt, 6, 2).
        gt_poses: list(np.array), img, (num_gt).

    Returns:

    """
    num_image = len(raw_results)
    det_corners, det_bboxes, det_poses = filter_results(
        raw_results, score_thr=0.35)
    if gt_bboxes is None:
        gt_bboxes = corners2bboxes(gt_corners)

    # fp, prec
    tpfpfn = [
        tpfp_func(det_bboxes[i], gt_bboxes[i], gt_ignores[i], iou_thr=0.4)
        for i in range(num_image)
    ]
    tp, fp, matched_indexs = tuple(zip(*tpfpfn))
    tp = np.hstack(tp)
    fp = np.hstack(fp)
    tp = sum(tp[0])
    fp = sum(fp[0])
    prec = tp / (tp + fp)

    # only one scale
    fn = []
    pose_tp = []
    vehicle_pose_tp = []
    corner_pairs = []
    for i in range(num_image):
        index_per_img = matched_indexs[i][0]
        gt_pose_per_img = gt_poses[i]
        gt_corners_per_img = gt_corners[i]
        gt_labels_per_img = gt_labels[i]
        if gt_ignores is not None:
            not_ignore = ~gt_ignores[i]
            index_per_img = index_per_img[not_ignore]
            gt_pose_per_img = gt_pose_per_img[not_ignore]
            gt_corners_per_img = gt_corners_per_img[not_ignore]
            gt_labels_per_img = gt_labels_per_img[not_ignore]

        fn.append(index_per_img == -1)
        if fn[-1].all():
            continue

        # ignore negtive pose
        pose_tp_per_img = det_poses[i][index_per_img] == gt_pose_per_img
        valid_pose = np.where(gt_pose_per_img >= 0)[0]
        pose_tp.append(pose_tp_per_img[valid_pose])
        vehicle_idx_per_img = find_vehicle(gt_labels_per_img)
        vehicle_pose_tp.append(
            pose_tp_per_img[list(set(valid_pose) & set(vehicle_idx_per_img))])

        # match corner based on pose
        for j in range(len(index_per_img)):
            p = gt_pose_per_img[j]
            corner_index = pose2index(p)
            gt_object_corners = gt_corners_per_img[j, corner_index, :]
            det_object_corners = det_corners[i][index_per_img[j],
                                                corner_index, :]
            corner_pairs.append(
                np.stack([det_object_corners, gt_object_corners], axis=2))

    # fn, recall
    fn = np.concatenate(fn)
    recall = 1 - sum(fn) / len(fn)

    # pose prec
    pose_tp = np.concatenate(pose_tp)
    pose_prec = sum(pose_tp) / len(pose_tp)
    vehicle_pose_tp = np.concatenate(vehicle_pose_tp)
    vehicle_pose_prec = sum(vehicle_pose_tp) / len(vehicle_pose_tp)
    print(len(pose_tp), len(vehicle_pose_tp))

    # corner err
    corner_pairs = np.vstack(corner_pairs)
    corner_err = np.sqrt(
        np.square(corner_pairs[:, 0, 0] - corner_pairs[:, 0, 1]) +
        np.square(corner_pairs[:, 1, 0] - corner_pairs[:, 1, 1])).mean()

    acc = dict(
        fn=sum(fn),
        fp=fp,
        corner_err=corner_err,
        pose_prec=pose_prec,
        vehicle_pose_prec=vehicle_pose_prec,
        prec=prec,
        recall=recall,
    )

    if display:
        print('\n-------corner acc------------')
        for k, v in acc.items():
            print('{:<10}:{:>8.3f}'.format(k, v))

    return acc