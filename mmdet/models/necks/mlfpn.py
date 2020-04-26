import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.utils import SEBlock, ConvModule


class MLFPN(nn.Module):

    def __init__(self,
                 in_channels=(512, 2048),
                 planes=256,
                 ffm_out_channels=(256, 512),
                 num_levels=8,
                 norm_cfg=dict(type='BN')):
        super(MLFPN, self).__init__()
        assert len(in_channels) == len(ffm_out_channels) == 2
        self.planes = planes
        self.num_levels = num_levels

        # construct base features
        self.reduce_s8 = ConvModule(in_channels[0], ffm_out_channels[0], 3,
                                    padding=1, norm_cfg=norm_cfg)
        self.reduce_s16 = ConvModule(in_channels[1], ffm_out_channels[1], 1, norm_cfg=norm_cfg)

        # share same params
        self.leach = nn.ModuleList(
            [ConvModule(sum(ffm_out_channels), self.planes // 2, 1, norm_cfg=norm_cfg)] *
            self.num_levels)

        for i in range(self.num_levels):
            side_channel = 512 if i == 0 else self.planes
            setattr(self, 'unet{}'.format(i + 1),
                    TUM(first_block=(i == 0),
                        input_planes=self.planes // 2,
                        is_smooth=self.smooth,
                        scales=self.num_scales,
                        side_channel=side_channel,
                        norm_cfg=norm_cfg))  # side channel isn't fixed.

        # construct SFAM module
        if self.sfam:
            self.sfam_module = SEBlock(self.planes, self.num_levels, self.num_scales,
                                       compress_ratio=16)

    def forward(self, x):
        """

        Args:
            x: tuple(tensor), len(x) == 2.

        Returns:

        """
        base_feature = torch.cat(
            (self.reduce_s8(x[0]),
             F.interpolate(self.reduce_s16(x[1]), scale_factor=2, mode='nearest')), 1)

        # tum_outs is the multi-level multi-scale feature
        tum_outs = []
        for i in range(self.num_levels):
            y = None if i == 0 else tum_outs[i - 1][-1]
            tum_outs.append(getattr(self, 'unet{}'.format(i + 1))(self.leach[i](base_feature), y))

        # concat with same scales
        sources = [torch.cat([fx[i - 1] for fx in tum_outs], 1)
                   for i in range(self.num_scales, 0, -1)]

        if self.sfam:
            sources = self.sfam_module(sources)
        return sources


class TUM(nn.Module):

    def __init__(self,
                 first_block=True,
                 input_planes=128,
                 is_smooth=True,
                 side_channel=512,
                 scales=6,
                 norm_cfg=dict(type='BN')):
        super(TUM, self).__init__()
        self.is_smooth = is_smooth
        self.first_block = first_block
        planes = 2 * input_planes
        in1 = input_planes if first_block else input_planes + side_channel

        self.layers = nn.Sequential()
        self.latlayer = nn.Sequential()
        self.toplayer = nn.Sequential(ConvModule(planes, planes, 1, norm_cfg=norm_cfg))

        # from shallow to deep
        self.layers.add_module('{}'.format(len(self.layers)),
                               ConvModule(in1, planes, 3, stride=2, padding=1, norm_cfg=norm_cfg))
        for i in range(scales - 2):
            stride, padding = (2, 1) if i != scales - 3 else (1, 0)
            self.layers.add_module(
                '{}'.format(len(self.layers)),
                ConvModule(planes, planes, 3, stride=stride, padding=padding, norm_cfg=norm_cfg))

        # from deep to shallow
        for i in range(scales - 2):
            self.latlayer.add_module('{}'.format(len(self.latlayer)),
                                     ConvModule(planes, planes, 3, norm_cfg=norm_cfg))
        self.latlayer.add_module('{}'.format(len(self.latlayer)),
                                 ConvModule(in1, planes, 3, norm_cfg=norm_cfg))

        if self.is_smooth:
            smooth = []
            for i in range(scales - 1):
                smooth.append(ConvModule(planes, planes, 1, norm_cfg=norm_cfg))
            self.smooth = nn.Sequential(*smooth)

    def _upsample_add(self, x, y, fuse_type='interp'):
        _, _, H, W = y.size()
        if fuse_type == 'interp':
            return F.interpolate(x, size=(H, W), mode='nearest') + y
        else:
            raise NotImplementedError

    def forward(self, x, y):
        x = x if self.first_block else torch.cat([x, y], 1)
        conved_feat = [x]
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            conved_feat.append(x)

        deconved_feat = [self.toplayer[0](conved_feat[-1])]
        for i in range(len(self.latlayer)):
            deconved_feat.append(self._upsample_add(
                deconved_feat[i], self.latlayer[i](conved_feat[len(self.layers) - 1 - i])))

        if self.is_smooth:
            # from deep to shallow
            smoothed_feat = [deconved_feat[0]]
            for i in range(len(self.smooth)):
                smoothed_feat.append(self.smooth[i](deconved_feat[i + 1]))
            return smoothed_feat

        return deconved_feat
