import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDetector
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result, tensor2imgs
from mmdet.core.utils.summary import (add_summary, add_image_summary,
                                      every_n_local_step, add_feature_summary)


@DETECTORS.register_module
class CenterNet2(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CenterNet2, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(CenterNet2, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_corners=None,
                      gt_classes=None,
                      gt_poses=None,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(
            img)  # each tensor in this tuple is corresponding to a level.
        outs = self.bbox_head(x)

        if gt_corners is None:
            loss_inputs = outs[:-1] + (gt_bboxes, gt_labels, img_metas,
                                       self.train_cfg)
            losses = self.bbox_head.loss_bboxes(*loss_inputs)
        else:
            loss_inputs = outs[:-1] + (gt_corners, gt_classes, gt_poses,
                                       img_metas, self.train_cfg)
            if hasattr(self.bbox_head, 'loss_corners'):
                losses = self.bbox_head.loss_corners(*loss_inputs)
            else:
                losses = self.bbox_head.loss(*loss_inputs)

        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        inputs = outs[:-1] + (img_meta, self.test_cfg, rescale)
        corners_list = self.bbox_head.get_corners(*inputs)

        bbox_results = [(c.cpu().numpy(), l.cpu().numpy(), s.cpu().numpy(),
                         p.cpu().numpy(), ps.cpu().numpy())
                        for c, l, s, p, ps in corners_list]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def dummy_forward(self, img, **kwargs):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # pose_scores, poses = F.softmax(outs[4], 1).max(1)
        poses = F.softmax(outs[4], 1).argmax(1).int()
        hm, idx = self.bbox_head.top_indexs(outs[0])
        raw_features = outs[-1].permute(0, 2, 3, 1).contiguous()
        return (hm, outs[1], outs[2], outs[3], poses, raw_features, idx)
