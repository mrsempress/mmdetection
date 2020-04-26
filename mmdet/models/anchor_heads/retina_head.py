import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob
from .anchor_head import AnchorHead
from mmdet.core.utils.summary import (add_summary, add_image_summary, every_n_local_step,
                                      add_feature_summary, add_histogram_summary, get_writer)


@HEADS.register_module
class RetinaHead(AnchorHead):
    """
    An anchor-based head used in [1]_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    References:
        .. [1]  https://arxiv.org/pdf/1708.02002.pdf

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes - 1)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 reg_in_channels=None,
                 reg_feat_channels=None,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        """Anchor head used in RetinaNet.

        :param num_classes: used to decide the channel num in cls branch.
        :param in_channels: input channel num.
        :param feat_channels: channel num of 4 conv.
        :param reg_in_channels: input channel num of reg branch, same as in_channels by default.
        :param reg_feat_channels: channel num of 4 conv of reg branch, same as feat_channels
            by default.
        :param stacked_convs: conv num stacked on the px feature map.
        :param octave_base_scale: 4 by default. Since the RetinaNet use the c3~c5 feature map
            in FPN, so the stride of px is [8, 16, 32, 64, 128]. In order to generate the anchor
            which has the stride of 32 to 512, we set octave_base_scale to 4.
        :param scales_per_octave: the author of RetinaNet added anchor of sizes
            {2^0, 2^(1/3), 2^(2/3)} for denser scale coverage.
        :param kwargs: see AnchorHead.
        """
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        self.reg_in_channels = reg_in_channels if reg_in_channels else in_channels
        self.reg_feat_channels = reg_feat_channels if reg_feat_channels else feat_channels
        super(RetinaHead, self).__init__(
            num_classes, in_channels,
            feat_channels=feat_channels, anchor_scales=anchor_scales, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # cls & reg branch do NOT share with each other.
        for i in range(self.stacked_convs):
            cls_chn = self.in_channels if i == 0 else self.feat_channels
            reg_chn = self.reg_in_channels if i == 0 else self.reg_feat_channels
            # The channel num of cls branch is also based on the class num, i.e. cls_out_channels.
            self.cls_convs.append(
                ConvModule(
                    cls_chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    reg_chn,
                    self.reg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.reg_feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single_level(self, x, idx):
        """
        Retina-R50: R50 takes 24.0ms, FPN takes 3.62ms, HEAD takes 17.48ms, consuming 46.5ms.

        |      | cls_feat | reg_feat | cls_score | bbox_pred | Total  |
        | ---- | -------- | -------- | --------- | --------- | ------ |
        | P3   | 3.47ms   | 3.46ms   | 2.23ms    | 0.24ms    | 9.41ms |
        | P4   | 1.30ms   | 1.26ms   | 0.74ms    | 0.11ms    | 3.43ms |
        | P5   | NA       | NA       | NA        | NA        | 1.92ms |
        | P6   | NA       | NA       | NA        | NA        | 1.35ms |
        | P7   | NA       | NA       | NA        | NA        | 1.37ms |

        Args:
            x: tensor.

        Returns:

        """
        # for a single level of multiply images.
        if isinstance(x, tuple):
            cls_feat, reg_feat = x[0], x[1]
        else:
            cls_feat, reg_feat = x, x

        if every_n_local_step():
            add_histogram_summary('retina_head_feat/cls_in_{}'.format(idx),
                                  cls_feat.detach().cpu())
            add_histogram_summary('retina_head_feat/reg_in_{}'.format(idx),
                                  reg_feat.detach().cpu())

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        if every_n_local_step():
            add_histogram_summary('retina_head_feat/cls_out_{}'.format(idx),
                                  cls_feat.detach().cpu())
            add_histogram_summary('retina_head_feat/reg_out_{}'.format(idx),
                                  reg_feat.detach().cpu())
            if idx == 0:
                for i, (cls_conv, reg_conv) in enumerate(zip(self.cls_convs, self.reg_convs)):
                    add_histogram_summary('retina_head_param/cls_conv_{}'.format(i),
                                          cls_conv.conv.weight.detach().cpu(), is_param=True)
                    add_histogram_summary('retina_head_param/reg_conv_{}'.format(i),
                                          reg_conv.conv.weight.detach().cpu(), is_param=True)

                add_histogram_summary('retina_head_param/cls_convs_grad',
                                      self.cls_convs[-1].conv.weight.grad.detach().cpu(),
                                      collect_type='none')
                add_histogram_summary('retina_head_param/reg_conv_grad',
                                      self.reg_convs[-1].conv.weight.grad.detach().cpu(),
                                      collect_type='none')

        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
