import os
import cv2
import numpy as np

side_colors = {
    'left': (0, 255, 222),
    'head': (100, 0, 222),
    'right': (20, 120, 200),
    'tail': (0, 200, 100),
}
side_corners = {
    'left': [0, 1, 4, 3],
    'right': [1, 2, 5, 4],
    'outer': [0, 2, 5, 3],
}
pose2sides = {
    0: [('tail', side_corners['outer'])],
    1: [('left', side_corners['left']), ('tail', side_corners['right'])],
    2: [('left', side_corners['outer'])],
    3: [('head', side_corners['left']), ('left', side_corners['right'])],
    4: [('head', side_corners['outer'])],
    5: [('right', side_corners['left']), ('head', side_corners['right'])],
    6: [('right', side_corners['outer'])],
    7: [('tail', side_corners['left']), ('right', side_corners['right'])],
}


def draw_polygon(img, corners_per_object, pose, pose_score=0):
    side_list = pose2sides[pose]
    x, y = corners_per_object[0, 0].astype(int)
    cv2.putText(img, '{:.2f}'.format(pose_score), (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    for s, idx in side_list:
        color = side_colors[s]
        vcorners = corners_per_object[idx, :]
        cv2.polylines(img, [vcorners], True, color, 2)


def show_corners(img,
                 results,
                 class_names,
                 score_thr=0.1,
                 out_file=None,
                 pad=0):
    corners, labels, scores, poses, pose_scores = results

    if pad > 0:
        h, w, c = img.shape
        shape = [h + pad, w + 2 * pad, c]
        new_img = np.ones(shape, dtype=img.dtype) * 255
        new_img[:h, pad:w + pad, :] = img
        img = new_img
        corners[..., 0] = corners[..., 0] + pad
    inds = np.where(scores > score_thr)[0]
    for i in inds.tolist():
        corners_per_object = corners[i].astype(np.int32).reshape(-1, 1, 2)
        draw_polygon(img, corners_per_object, poses[i], scores[i])

    if out_file is not None:
        dirname = os.path.dirname(out_file)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)
        cv2.imwrite(out_file, img)
    else:
        return img
