import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.models.utils import yolov3_conv2d
from ..registry import NECKS


@NECKS.register_module
class DarknetFPN(nn.Module):

    def __init__(self,
                 in_channels=(256, 512, 1024),
                 out_channels=(256, 512, 1024),
                 norm_cfg=None,
                 norm_eval=True):
        super(DarknetFPN, self).__init__()
        self.norm_eval = norm_eval

        blocks, transitions = [], []
        # the output layers are used in reverse order.
        for i, (in_channel, channel) in enumerate(zip(in_channels[::-1], out_channels[::-1])):
            if i > 0:
                trans_incha = out_channels[-i] // 2  # 512, 256
                trans_outcha = out_channels[-i - 1] // 2  # 256, 128
                in_channel += trans_outcha
            blocks.append(DarknetDetectionBlock(in_channel, channel, norm_cfg=norm_cfg))

            if i > 0:
                transitions.append(yolov3_conv2d(trans_incha, trans_outcha, 1, 0, 1,
                                                 norm_cfg=norm_cfg))

        self.darknet_fpn_block = nn.Sequential(*blocks)
        self.transitions = nn.Sequential(*transitions)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, feats):
        """

        Args:
            feats: list(tensor), tensor <=> image.

        Returns:
            out: list(tensor) <=> [lv3, lv4, lv5].
        """
        out = []
        x = feats[-1]
        # the YOLO output layers are used in reverse order, i.e., from very deep layers to shallow.
        for i, block in enumerate(self.darknet_fpn_block):
            x, tip = block(x)
            out.append(tip)
            if i >= len(feats) - 1:
                break

            # use transition layer to the output of detection block of deep layer.
            x = self.transitions[i](x)
            # upsample feature map reverse to shallow layers
            upsample = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            feat_now = feats[::-1][i + 1]
            x = torch.cat((upsample[:, :, :feat_now.size(2), :feat_now.size(3)], feat_now), dim=1)
        return out[::-1]

    def train(self, mode=True):
        super(DarknetFPN, self).train(mode)
        if mode and self.norm_eval:
            for _, m in self.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


class DarknetDetectionBlock(nn.Module):
    """Basic block for darknet detection."""

    def __init__(self,
                 in_channel,
                 out_channel,
                 norm_cfg):
        super(DarknetDetectionBlock, self).__init__()

        body = []
        for j in range(2):
            incha = in_channel if j == 0 else out_channel
            body.extend([
                yolov3_conv2d(incha, out_channel // 2, 1, 0, 1, norm_cfg=norm_cfg),
                yolov3_conv2d(out_channel // 2, out_channel, 3, 1, 1, norm_cfg=norm_cfg)
            ])
        body.extend([
            yolov3_conv2d(out_channel, out_channel // 2, 1, 0, 1, norm_cfg=norm_cfg)
        ])

        self.body = nn.Sequential(*body)
        self.tip = nn.Sequential(
            yolov3_conv2d(out_channel // 2, out_channel, 3, 1, 1, norm_cfg=norm_cfg)
        )

    def forward(self, x):
        route = self.body(x)
        tip = self.tip(route)
        return route, tip
