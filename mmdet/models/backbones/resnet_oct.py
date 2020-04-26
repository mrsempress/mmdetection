import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.plugins import GeneralizedAttention
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from mmdet.core.utils.summary import add_histogram_summary, every_n_local_step
from ..registry import BACKBONES
from ..utils import (
    build_conv_layer, build_norm_layer, CBAMBlock, MaskModule, SEBlock, ShareConv2d, ShareBN)



class OctBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=dict(type='ShareConv'),
                 norm_cfg=dict(type='ShareBN'),
                 bias=False,
                 use_spatial_attention=False,
                 use_se_block=False,
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        super(OctBasicBlock, self).__init__()
        assert gen_attention is None, "Not implemented yet."
        assert gcb is None, "Not implemented yet."

        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.use_spatial_attention = use_spatial_attention
        self.use_se_block = use_se_block

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        alpha_out = conv_cfg.pop('alpha_out', None) if conv_cfg is not None else None

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=bias)
        self.add_module(self.norm1_name, norm1)

        if conv_cfg is not None:
            conv_cfg['alpha_out'] = alpha_out
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=bias)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        if isinstance(out, tuple):
            out = (self.relu(out[0]), self.relu(out[1]))
        else:
            out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            if isinstance(x, tuple):
                x_predown = torch.cat([x[0], F.upsample(x[1], scale_factor=2)], dim=1)
                identity = self.downsample(x_predown)
            else:
                identity = self.downsample(x)

        if isinstance(out, tuple):
            if isinstance(identity, tuple):
                out = (out[0] + identity[0], out[1] + identity[1])
            else:
                out = (out[0] + identity[:, :out[0].size(1)],
                       out[1] + F.avg_pool2d(identity[:, -out[1].size(1):], 2))
            out = (self.relu(out[0]), self.relu(out[1]))
        else:
            if isinstance(identity, tuple):
                identity = torch.cat([identity[0], F.upsample(identity[1], scale_factor=2)], dim=1)
            out += identity
            out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 bias=False,
                 use_spatial_attention=False,
                 use_se_block=False,
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert gcb is None or isinstance(gcb, dict)
        assert gen_attention is None or isinstance(gen_attention, dict)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.gcb = gcb
        self.with_gcb = gcb is not None
        self.gen_attention = gen_attention
        self.with_gen_attention = gen_attention is not None

        assert not use_spatial_attention, NotImplementedError
        self.use_se_block = use_se_block

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=bias)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=bias)
        else:
            assert conv_cfg is None, 'conv_cfg must be None for DCN'
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18  # 2 offsets for each position (3x3).
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27  # 2 offsets + 1 scale for each position (3x3).
            self.conv2_offset = nn.Conv2d(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=bias)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=bias)
        self.add_module(self.norm3_name, norm3)

        if use_se_block:
            self.se_block = SEBlock(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_gcb:
            gcb_inplanes = planes * self.expansion
            self.context_block = ContextBlock(inplanes=gcb_inplanes, **gcb)

        # gen_attention
        if self.with_gen_attention:
            self.gen_attention_block = GeneralizedAttention(
                planes, **gen_attention)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if not self.with_dcn:
                out = self.conv2(out)
            elif self.with_modulated_dcn:
                offset_mask = self.conv2_offset(out)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                out = self.conv2(out, offset, mask)
            else:
                offset = self.conv2_offset(out)
                out = self.conv2(out, offset)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_gen_attention:
                out = self.gen_attention_block(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_gcb:
                out = self.context_block(out)

            if self.use_se_block:
                out = self.se_block(out) * out

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   bias=False,
                   use_spatial_attention=False,
                   use_se_block=False,
                   use_trident_c4=False,
                   use_trident_c5=False,
                   last_normal=False,
                   dcn=None,
                   gcb=None,
                   gen_attention=None,
                   gen_attention_blocks=[]):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = [
            build_conv_layer(
                dict(type='Conv'),
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=bias),
            build_norm_layer(norm_cfg, planes * block.expansion)[1]]
        if use_se_block:
            downsample.append(SEBlock(planes * block.expansion))

        downsample = nn.Sequential(*downsample)

    layers = []
    layers.append(
        block(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias,
            use_spatial_attention=use_spatial_attention,
            use_se_block=use_se_block,
            dcn=dcn,
            gcb=gcb,
            gen_attention=gen_attention if
            (0 in gen_attention_blocks) else None))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        if last_normal and i == blocks - 1:
            conv_cfg['alpha_out'] = 0
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=1,
                dilation=dilation,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                bias=bias,
                use_spatial_attention=use_spatial_attention,
                use_se_block=use_se_block,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention if
                (i in gen_attention_blocks) else None))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNetOct(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings = {
        18: (OctBasicBlock, (2, 2, 2, 2)),
        34: (OctBasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 oct_start_layer=2,
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 local_conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 bias=False,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 stage_with_local_conv_cfg=(False, False, False, False),
                 stage_with_se=(False, False, False, False),
                 use_spatial_attention=False,
                 attention_mask_layer_n=None,
                 use_trident_c4=False,
                 use_trident_c5=False,
                 add_summay_every_n_step=None,
                 gcb=None,
                 stage_with_gcb=(False, False, False, False),
                 gen_attention=None,
                 stage_with_gen_attention=((), (), (), ()),
                 with_cp=False,
                 zero_init_residual=True):
        super(ResNetOct, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert 4 >= num_stages >= 1
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.bias = bias
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.gen_attention = gen_attention
        self.gcb = gcb
        self.stage_with_gcb = stage_with_gcb
        if gcb is not None:
            assert len(stage_with_gcb) == num_stages
        if local_conv_cfg is not None:
            assert len(stage_with_local_conv_cfg) == num_stages
        assert attention_mask_layer_n is None or 1 <= attention_mask_layer_n <= 4
        self.attention_mask_layer_n = attention_mask_layer_n
        assert not (use_trident_c4 and use_trident_c5)
        self.add_summay_every_n_step = add_summay_every_n_step
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            gcb = self.gcb if self.stage_with_gcb[i] else None
            planes = 64 * 2 ** i
            conv_cfg = self.conv_cfg if not stage_with_local_conv_cfg[i] else local_conv_cfg
            with_se = stage_with_se[i]
            if i >= oct_start_layer:
                conv_cfg = dict(type='ShareConv')
                norm_cfg = dict(type='ShareBN')
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                bias=bias,
                use_spatial_attention=use_spatial_attention,
                use_se_block=with_se,
                use_trident_c4=use_trident_c4 if i == 2 else False,
                use_trident_c5=use_trident_c5 if i == 3 else False,
                last_normal=(i == len(self.stage_blocks) - 1),
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention,
                gen_attention_blocks=stage_with_gen_attention[i])
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * 64 * 2 ** (len(self.stage_blocks) - 1)

        if attention_mask_layer_n:
            mask_stride = 4
            for i in range(attention_mask_layer_n):
                mask_stride *= strides[i]
            self.attention_mask = MaskModule(
                int(64 * 2 ** (attention_mask_layer_n - 1) * self.block.expansion),
                feat_stride=mask_stride)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=self.bias)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, (OctBasicBlock, Bottleneck)) and hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, OctBasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        # R50: 22.5ms
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        mask = None
        for i, layer_name in enumerate(self.res_layers):
            if self.attention_mask_layer_n and self.attention_mask_layer_n == i:
                x, mask = self.attention_mask(x)
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)

            if self.add_summay_every_n_step and every_n_local_step(self.add_summay_every_n_step):
                add_histogram_summary('resnet_feat_layer{}'.format(i + 1), x.detach().cpu())
                add_histogram_summary('resnet_weight_layer{}'.format(i + 1),
                                      res_layer[-1].conv2.weight.detach().cpu(), is_param=True)

        if self.attention_mask_layer_n:
            return tuple(outs), mask

        return tuple(outs)

    def train(self, mode=True):
        super(ResNetOct, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for _, m in self.named_modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
