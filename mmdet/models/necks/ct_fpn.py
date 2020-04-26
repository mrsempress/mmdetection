import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.ops import ModulatedDeformConvPack
from mmdet.models.utils import ConvModule, build_norm_layer
from ..registry import NECKS


@NECKS.register_module
class CTFPN(nn.Module):

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 outplane=64,
                 shortcut_kernel=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 norm_eval=False):
        super(CTFPN, self).__init__()
        self.norm_eval = norm_eval

        # repeat deconv 3 times, 32x to 4x.
        self.deconv_layers = nn.ModuleList()
        self.shortcut_layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()
        self.deconv_layers.extend(
            [self._make_deconv_layer(inplanes[-1], 1, [planes[0]], [4], norm_cfg=norm_cfg),
             self._make_deconv_layer(planes[0], 1, [planes[1]], [4], norm_cfg=norm_cfg),
             self._make_deconv_layer(planes[1], 1, [planes[2]], [4], norm_cfg=norm_cfg)])

        padding = (shortcut_kernel - 1) // 2
        self.shortcut_layers.extend(
            [ConvModule(inplanes[2], planes[0], shortcut_kernel,
                        padding=padding, conv_cfg=conv_cfg, activation=None),
             ConvModule(inplanes[1], planes[1], shortcut_kernel,
                        padding=padding, conv_cfg=conv_cfg, activation=None),
             ConvModule(inplanes[0], planes[2], shortcut_kernel,
                        padding=padding, conv_cfg=conv_cfg, activation=None)])

        self.output_layers.extend(
            [ConvModule(planes[0], outplane, 1, conv_cfg=conv_cfg, activation=None),
             ConvModule(planes[1], outplane, 1, conv_cfg=conv_cfg, activation=None),
             ConvModule(planes[2], outplane, 1, conv_cfg=conv_cfg, activation=None)])

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding, output_padding = (1, 0)
        elif deconv_kernel == 3:
            padding, output_padding = (1, 1)
        elif deconv_kernel == 2:
            padding, output_padding = (0, 0)
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, inplanes, num_layers, num_filters, num_kernels, norm_cfg=None):
        """

        Args:
            inplanes: in-channel num.
            num_layers: deconv layer num.
            num_filters: out channel of the deconv layers.
            num_kernels: int
            norm_cfg: dict()

        Returns:
            stacked deconv layers.
        """
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            inplanes = inplanes if i == 0 else num_filters[i - 1]

            mdcn = ModulatedDeformConvPack(inplanes, planes, 3, stride=1,
                                           padding=1, dilation=1, deformable_groups=1)
            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            self.fill_up_weights(up)

            layers.append(mdcn)
            if norm_cfg:
                layers.append(build_norm_layer(norm_cfg, planes)[1])
            layers.append(nn.ReLU(inplace=True))

            layers.append(up)
            if norm_cfg:
                layers.append(build_norm_layer(norm_cfg, planes)[1])
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def fill_up_weights(self, up):
        w = up.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feats):
        """
        Args:
            feats: list(tensor), tensor <=> image. [lv2, lv3, lv4, lv5]

        Returns:
            out: list(tensor). [lv2, lv3, lv4].
        """
        x = feats[-1]
        output = []
        for i, (deconv_layer, shortcut_layer, output_layer) in enumerate(
                zip(self.deconv_layers, self.shortcut_layers, self.output_layers)):
            x = deconv_layer(x)
            shortcut = shortcut_layer(feats[-i - 2])
            x = x + shortcut
            output.append(output_layer(x))
        output = output[::-1]
        return output

    def train(self, mode=True):
        super(CTFPN, self).train(mode)
        if mode and self.norm_eval:
            for _, m in self.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
