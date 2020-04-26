import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init, constant_init

from mmdet.ops import ModulatedDeformConvPack
from mmdet.models.utils import (
    ConvModule, build_conv_layer, build_norm_layer, bias_init_with_prob)
from ..registry import NECKS


class UpsamplingLayers(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 no_upsample=False,
                 vallina=False,
                 offset_mean=False):
        if vallina:
            if isinstance(vallina, int):
                padding = int((vallina - 1) / 2)
                dila = padding
                mdcn = nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    padding=padding,
                    dilation=dila)
            else:
                mdcn = nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    padding=1)
        elif conv_cfg:
            mdcn = build_conv_layer(conv_cfg, in_channels, out_channels)
        else:
            mdcn = ModulatedDeformConvPack(
                in_channels,
                out_channels,
                3,
                offset_mean=offset_mean,
                stride=1,
                padding=1,
                dilation=1,
                deformable_groups=1)
        layers = []
        layers.append(mdcn)
        if norm_cfg:
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        layers.append(nn.ReLU(inplace=True))
        if not no_upsample:
            layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
        super(UpsamplingLayers, self).__init__(*layers)


class ShortcutConnection(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, conv_cfg,
                 down=False, down_sec=False):
        super(ShortcutConnection, self).__init__()
        layers = []
        for i, kernel_size in enumerate(kernel_sizes):
            inc = in_channels if i == 0 else out_channels
            stride = 1
            if (down and i == 0) or (down_sec and i == 1):
                stride = 2
            padding = (kernel_size - 1) // 2
            if conv_cfg:
                layers.append(
                    build_conv_layer(conv_cfg, inc, out_channels, kernel_size,
                                     stride=stride, padding=padding))
            else:
                layers.append(
                    nn.Conv2d(inc, out_channels, kernel_size,
                              stride=stride, padding=padding))
            if i < len(kernel_sizes) - 1:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


@NECKS.register_module
class TTFXFPN(nn.Module):

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 shortcut_cfg=(1, 2, 3),
                 use_extra_shortcut=True,
                 s16_shortcut_twice=False,
                 shortcut_conv_cfg=None,
                 up_conv_cfg=None,
                 upsample_vallina=False,
                 dcn_offset_mean=False,
                 down_ratio=(8, 4),
                 fake_s8=False,
                 norm_cfg=dict(type='BN')):
        super(TTFXFPN, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.shortcut_cfg = shortcut_cfg
        self.use_extra_shortcut = use_extra_shortcut
        self.s16_shortcut_twice = s16_shortcut_twice
        self.shortcut_conv_cfg = shortcut_conv_cfg
        self.up_conv_cfg = up_conv_cfg
        self.upsample_vallina = upsample_vallina
        self.dcn_offset_mean = dcn_offset_mean
        assert len(down_ratio) in [1, 2, 3]
        self.down_ratio= down_ratio
        self.fake_s8 = fake_s8
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_extra_layers(self):
        self.s8_upsample_layers = nn.ModuleList()
        self.s8_shortcut_layers = nn.ModuleList()
        self.s16_upsample_layers = nn.ModuleList()
        self.s16_shortcut_layers = nn.ModuleList()

        if 8 in self.down_ratio:
            self.s8_upsample_layers.append(
                UpsamplingLayers(self.planes[1], self.planes[1],
                                 conv_cfg=self.up_conv_cfg,
                                 norm_cfg=self.norm_cfg, no_upsample=not self.fake_s8,
                                 vallina=self.upsample_vallina, offset_mean=self.dcn_offset_mean))
            self.s8_shortcut_layers.append(
                ShortcutConnection(
                    self.inplanes[0], self.planes[1], [3] * self.shortcut_cfg[2],
                    self.shortcut_conv_cfg, down=not self.fake_s8))
        if 16 in self.down_ratio:
            self.s16_upsample_layers.extend([
                UpsamplingLayers(self.planes[0], self.planes[0],
                                 conv_cfg=self.up_conv_cfg,
                                 norm_cfg=self.norm_cfg, no_upsample=True,
                                 vallina=self.upsample_vallina, offset_mean=self.dcn_offset_mean),
                UpsamplingLayers(self.planes[0], self.planes[0],
                                 conv_cfg=self.up_conv_cfg,
                                 norm_cfg=self.norm_cfg, no_upsample=True,
                                 vallina=self.upsample_vallina, offset_mean=self.dcn_offset_mean),
            ])
            self.s16_shortcut_layers.append(
                ShortcutConnection(
                    self.inplanes[1], self.planes[0], [3] * self.shortcut_cfg[1],
                    self.shortcut_conv_cfg, down=True))
            if self.s16_shortcut_twice:
                self.s16_shortcut_layers.append(
                    ShortcutConnection(
                        self.inplanes[0], self.planes[0], [3] * self.shortcut_cfg[2],
                        self.shortcut_conv_cfg, down=True, down_sec=True))

    def _init_layers(self):
        self.upsample_layers = nn.ModuleList([
            UpsamplingLayers(
                self.inplanes[-1], self.planes[0], conv_cfg=self.up_conv_cfg,
                norm_cfg=self.norm_cfg, vallina=self.upsample_vallina,
                offset_mean=self.dcn_offset_mean),
            UpsamplingLayers(
                self.planes[0], self.planes[1], conv_cfg=self.up_conv_cfg,
                norm_cfg=self.norm_cfg, vallina=self.upsample_vallina,
                offset_mean=self.dcn_offset_mean),
            UpsamplingLayers(
                self.planes[1], self.planes[2], conv_cfg=self.up_conv_cfg,
                norm_cfg=self.norm_cfg, vallina=self.upsample_vallina,
                offset_mean=self.dcn_offset_mean)
        ])

        self.shortcut_layers = nn.ModuleList([
            ShortcutConnection(self.inplanes[2], self.planes[0], [3] * self.shortcut_cfg[0],
                               self.shortcut_conv_cfg),
            ShortcutConnection(self.inplanes[1], self.planes[1], [3] * self.shortcut_cfg[1],
                               self.shortcut_conv_cfg),
            ShortcutConnection(self.inplanes[0], self.planes[2], [3] * self.shortcut_cfg[2],
                               self.shortcut_conv_cfg)
        ])
        self._init_extra_layers()

    def init_weights(self):
        for upsample_layers in [self.upsample_layers, self.s8_upsample_layers,
                                self.s16_upsample_layers]:
            for m in upsample_layers.modules():
                if isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

        for shortcut_layers in [self.shortcut_layers, self.s8_shortcut_layers,
                                self.s16_shortcut_layers]:
            for m in shortcut_layers.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)

        for m in self.modules():
            if isinstance(m, ModulatedDeformConvPack):
                constant_init(m.conv_offset, 0)

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            out: list(tensor).
        """
        y, shortcuts = [], []
        x = feats[-1]
        for i, shortcut_layer in enumerate(self.shortcut_layers):
            shortcuts.append(shortcut_layer(feats[-i - 2]))

        for i, upsampling_layer in enumerate(self.upsample_layers):
            x = upsampling_layer(x)
            x = x + shortcuts[i]
            y.append(x)

        out = []
        if 16 in self.down_ratio:
            y_s16 = y[0]
            y_s16 = self.s16_upsample_layers[0](y_s16)
            if self.use_extra_shortcut:
                y_s16 = y_s16 + self.s16_shortcut_layers[0](feats[1])
            y_s16 = self.s16_upsample_layers[1](y_s16)
            if self.s16_shortcut_twice:
                y_s16 = y_s16 + self.s16_shortcut_layers[1](feats[0])
            out.append(y_s16)

        if 8 in self.down_ratio:
            y_s8 = y[1]
            y_s8 = self.s8_upsample_layers[0](y_s8)
            if self.use_extra_shortcut:
                y_s8 = y_s8 + self.s8_shortcut_layers[0](feats[0])
            out.append(y_s8)

        if 4 in self.down_ratio:
            y_s4 = y[-1]
            out.append(y_s4)
        out = out[::-1]
        return out
