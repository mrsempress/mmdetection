import numpy as np
import cv2

__all__ = ['draw_boxes', 'draw_text']


def draw_boxes(im, boxes=None, labels=None, color=None):
    """
    Args:
        im (np.ndarray): a BGR image in range [0,255]. It will not be modified.
        boxes (np.ndarray): a numpy array of shape Nx4 where each row is [x1, y1, x2, y2].
        labels: (list[str] or None)
        color: a 3-tuple BGR color (in range [0, 255])

    Returns:
        np.ndarray: a new image.
    """
    im = im.copy()
    if color is None:
        color = (15, 128, 15)
    if im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    if boxes is not None:
        boxes = np.asarray(boxes, dtype='int32')
        if labels:
            assert len(labels) == len(boxes), "{} != {}".format(len(labels), len(boxes))
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        sorted_inds = np.argsort(-areas)  # draw large ones first
        assert areas.min() > 0, areas.min()
        # allow equal, because we are not very strict about rounding error here
        assert boxes[:, 0].min() >= 0 and boxes[:, 1].min() >= 0 \
               and boxes[:, 2].max() <= im.shape[1] and boxes[:, 3].max() <= im.shape[0], \
            "Image shape: {}\n Boxes:\n{}".format(str(im.shape), str(boxes))

        for i in sorted_inds:
            box = boxes[i, :]
            if labels is not None:
                im = draw_text(im, (box[0], box[1]), str(labels[i]), color=color)
            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]),
                          color=color, thickness=2)
    return im


def draw_text(img, pos, text, color, font_scale=0.8):
    """
    Draw text on an image.

    Args:
        pos (tuple): x, y; the position of the text
        text (str):
        font_scale (float):
        color (tuple): a 3-tuple BGR color in [0, 255]
    """
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, 1)
    # Place text background.
    if x0 + text_w > img.shape[1]:
        x0 = img.shape[1] - text_w
    if y0 - int(1.15 * text_h) < 0:
        y0 = int(1.15 * text_h)
    back_topleft = x0, y0 - int(1.3 * text_h)
    back_bottomright = x0 + text_w, y0
    cv2.rectangle(img, back_topleft, back_bottomright, color, -1)
    # Show text.
    text_bottomleft = x0, y0 - int(0.25 * text_h)
    cv2.putText(img, text, text_bottomleft, font, font_scale, (222, 222, 222), lineType=cv2.LINE_AA)
    return img
