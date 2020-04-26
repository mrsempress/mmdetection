import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.utils.attention_module import CBAMBlock
from mmdet.core.utils.summary import (
    get_epoch, get_total_epoch, get_inner_iter, get_total_inner_iter, add_summary)
from mmdet.models.utils.norm import ShareBN

_dyn_dila_count = 0


class SeparableConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 kernels_per_layer=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels * kernels_per_layer,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer,
                                   out_channels,
                                   kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class TridentConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 group_3x3=3):
        super(TridentConv2d, self).__init__()
        assert group_3x3 >= 1
        self.group_3x3 = group_3x3

        group = group_3x3 + 1  # 1x1 + 3x3-dila-1 + 3x3-dila-2 + ...
        out = int(out_channels * 2 / group)
        self.conv_1x1 = nn.Conv2d(in_channels, out, 1)
        self.weight = nn.Parameter(torch.randn(out, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out))
        nn.init.normal_(self.weight, 0, 0.01)

        self.bottleneck = nn.Conv2d(out * group, out_channels, 1)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_fuse = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        out = []
        out.append(self.conv_1x1(x))
        for i in range(1, self.group_3x3 + 1):
            out.append(F.conv2d(x, self.weight, bias=self.bias, padding=i, dilation=i))

        out = torch.cat(out, dim=1)
        out = F.relu(out)

        out = self.bottleneck(out)
        shortcut = self.conv_shortcut(x)
        out = out + shortcut

        out = self.conv_fuse(out)
        out = F.relu(out)
        return out


class MultiScaleConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(MultiScaleConv2d, self).__init__()
        assert out_channels % 4 == 0
        outc = int(out_channels / 4)

        self.conv3_d1 = nn.Conv2d(in_channels, outc, 3, padding=1)
        self.conv3_d2 = nn.Conv2d(in_channels, outc, 3, padding=2, dilation=2)
        self.conv3_d3 = nn.Conv2d(in_channels, outc, 3, padding=3, dilation=3)
        self.conv3_d4 = nn.Conv2d(in_channels, outc, 3, padding=4, dilation=4)

    def forward(self, x):
        out_d1 = self.conv3_d1(x)
        out_d2 = self.conv3_d2(x)
        out_d3 = self.conv3_d3(x)
        out_d4 = self.conv3_d4(x)

        out = torch.cat([out_d1, out_d2, out_d3, out_d4], dim=1)
        out = F.relu(out)
        return out


class ExpConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 neg_x=False):
        super(ExpConv2d, self).__init__()
        self.neg_x = neg_x
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = x.exp()
        if self.neg_x:
            x = -x
        return self.conv(x)


class ShortcutConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 paddings,
                 activation_last=False,
                 use_prelu=False,
                 use_cbam=False,
                 shortcut_in_shortcut=False):
        super(ShortcutConv2d, self).__init__()
        assert len(kernel_sizes) == len(paddings)
        self.shortcut_in_shortcut = shortcut_in_shortcut

        layers = []
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes, paddings)):
            inc = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(inc, out_channels, kernel_size, padding=padding))
            if i < len(kernel_sizes) - 1 or activation_last:
                if use_prelu:
                    layers.append(nn.PReLU(out_channels))
                else:
                    layers.append(nn.ReLU(inplace=True))

        if use_cbam:
            layers.append(CBAMBlock(planes=out_channels))

        self.layers = nn.Sequential(*layers)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        y = self.layers(x)
        if self.shortcut_in_shortcut:
            shortcut = self.shortcut(x)
            y = shortcut + y
        return y


class SymConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 direction='h',
                 merge='all',
                 *args,
                 bias=True,
                 **kwargs):
        super(SymConv2d, self).__init__()
        assert kernel_size >= 3
        assert direction in ('h', 'v')
        assert merge in ('all', 'top-left', 'bottom-right')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = (kernel_size - 1) // 2
        self.dilation = 1
        self.transposed = False
        self.output_padding = 0
        self.groups = 1

        self.direction = direction
        self.merge = merge
        self.use_bias = bias

        kernel_sizes = (kernel_size, 1) if direction == 'h' else (1, kernel_size)
        paddings = (self.padding, 0) if direction == 'h' else (0, self.padding)
        self.conv_middle = nn.Conv2d(in_channels, out_channels, kernel_sizes,
                                     padding=paddings, bias=False)
        self.conv_margins = nn.ModuleList()
        for i in range(self.padding):
            paddings = (self.padding, i + 1) if direction == 'h' else (i + 1, self.padding)
            self.conv_margins.append(
                nn.Conv2d(in_channels, out_channels, kernel_sizes, padding=paddings, bias=False)
            )
        nn.init.normal_(self.conv_middle.weight, 0, 0.01)
        for m in self.conv_margins:
            nn.init.normal_(m.weight, 0, 0.01)
        self.weight = nn.Parameter(torch.zeros(1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, x):
        x_middle = self.conv_middle(x)
        x_margins = []
        for conv_margin in self.conv_margins:
            x_margins.append(conv_margin(x))
        if self.direction == 'h':
            x_margin = x_margins[0][:, :, :, 0:-2] + x_margins[0][:, :, :, 2:]
            for i, xm in enumerate(x_margins[1:], start=1):
                if self.merge == 'all':
                    x_margin += (xm[:, :, :, 0:-i * self.padding * 2] +
                                 xm[:, :, :, i * self.padding * 2:])
                elif self.merge == 'top-left':
                    x_margin += xm[:, :, :, 0:-i * self.padding * 2]
                else:
                    x_margin += xm[:, :, :, i * self.padding * 2:]
        else:
            x_margin = x_margins[0][:, :, 0:-2] + x_margins[0][:, :, 2:]
            for i, xm in enumerate(x_margins[1:], start=1):
                if self.merge == 'all':
                    x_margin += (xm[:, :, 0:-i * self.padding * 2] +
                                 xm[:, :, i * self.padding * 2:])
                elif self.merge == 'top-left':
                    x_margin += xm[:, :, 0:-i * self.padding * 2]
                else:
                    x_margin += xm[:, :, i * self.padding * 2:]

        y = x_middle + x_margin
        if self.use_bias:
            return y + self.bias
        return y


class BasicBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.norm1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(planes)

        if inplanes != planes or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)

        if hasattr(self, 'downsample'):
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)

        return out


class BasicBlock2(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1):
        super(BasicBlock2, self).__init__()
        self.block1 = BasicBlock(inplanes, planes, stride, dilation)
        self.block2 = BasicBlock(planes, planes, 1, 1)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out


class WHSymConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 *args,
                 **kwargs):
        super(WHSymConv2d, self).__init__()
        assert out_channels == 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = (kernel_size - 1) // 2
        self.dilation = 1
        self.transposed = False
        self.output_padding = 0
        self.groups = 1

        self.wh_convs = nn.ModuleList([
            SymConv2d(in_channels, 1, kernel_size, *args,
                      direction='h', merge='top-left', **kwargs),
            SymConv2d(in_channels, 1, kernel_size, *args,
                      direction='v', merge='top-left', **kwargs),
            SymConv2d(in_channels, 1, kernel_size, *args,
                      direction='h', merge='bottom-right', **kwargs),
            SymConv2d(in_channels, 1, kernel_size, *args,
                      direction='v', merge='bottom-right', **kwargs),
        ])

        self.weight = nn.Parameter(torch.zeros(1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, x):
        feats = []
        for m in self.wh_convs:
            feats.append(m(x))
        return torch.cat(feats, dim=1)


class ShareConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 alpha_out=None,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        super(ShareConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.alpha_out = nn.Parameter(torch.zeros(1, )) if alpha_out is None else alpha_out

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels, ))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, alpha=None):
        x_h, x_l = x if type(x) is tuple else (x, None)
        if not isinstance(self.alpha_out, nn.Parameter):
            self.alpha_out = torch.tensor(self.alpha_out)
        alpha_out = self.alpha_out if alpha is None else alpha

        x_h2l = None
        inc_h = x_h.size(1)
        outc_h = (self.out_channels * (1 - alpha_out)).int()

        x_h = x_h if self.stride == 1 else self.downsample(x_h)
        if isinstance(self.alpha_out, nn.Parameter):
            alpha_out = (0.5 - 0.5 * F.sigmoid(self.alpha_out))
            if alpha_out > 0:
                outc_h = (self.out_channels * (1 - alpha_out)).int()
                bias_l = self.bias[outc_h:] if self.bias is not None else None
                x_h2l = F.conv2d(self.downsample(x_h),
                                 self.weight[outc_h:, :inc_h], bias_l, 1, 1, 1, 1)
        bias_h = self.bias[:outc_h] if self.bias is not None else None
        x_h2h = F.conv2d(x_h, self.weight[:outc_h, :inc_h], bias_h, 1, 1, 1, 1)

        if x_l is not None:
            bias_l = self.bias[:outc_h] if self.bias is not None else None
            inc_l = x_l.size(1)
            x_l2h = F.conv2d(x_l, self.weight[:outc_h, -inc_l:], bias_h, 1, 1, 1, 1)
            x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h

            x_l2l = None
            if alpha_out > 0:
                x_l2l = x_l if self.stride == 1 else self.downsample(x_l)
                x_l2l = F.conv2d(x_l2l, self.weight[outc_h:, -inc_l:], bias_l, 1, 1, 1, 1)
            x_h = x_l2h + x_h2h
            x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
            return x_h, x_l
        else:
            return x_h2h, x_h2l


class DynDialConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 alpha_out=None,
                 use_beta=False,
                 use_bn=False,
                 use_split=True,
                 auto_lr=False,
                 init_values=None,
                 dilation_choice=(1, 2, 3),
                 alpha_scale=1e-4,
                 param_multiply=1.,
                 param_beta_multiply=1.,
                 tau=(10, 1),
                 tau_start_epoch=None,
                 tau_end_epoch=None,
                 sin_tau=False,
                 end_after_end=False,
                 fixed_params_dict=dict(),
                 split=16,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        super(DynDialConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert kernel_size in (1, 3)
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        assert out_channels % split == 0
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.param_multiply = param_multiply  # to enlarge the gap after softmax
        self.param_beta_multiply = param_beta_multiply
        self.use_split = use_split
        self.dilation_choice = dilation_choice
        self.tau = tau
        self.tau_start_epoch = tau_start_epoch if tau_start_epoch else 1
        self.tau_end_epoch = tau_end_epoch
        self.sin_tau = sin_tau
        self.end_after_end = end_after_end
        self.split = split
        self.auto_lr = auto_lr
        self.use_bn = use_bn
        global _dyn_dila_count
        self.count = _dyn_dila_count
        _dyn_dila_count += 1
        self.fixed_param = fixed_params_dict.get(self.count, None)
        if self.fixed_param:
            assert use_split
            self.fixed_param = torch.tensor(self.fixed_param)

        if kernel_size == 1:
            self.weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_channels, ))
            else:
                self.register_parameter('bias', None)
        else:
            x_shape = (split,) if use_split else (split, len(dilation_choice))
            if init_values is None:
                x = torch.randn(x_shape) * alpha_scale
            else:
                assert len(init_values) == x_shape[-1]
                x = torch.tensor(init_values).log() * alpha_scale
                assert len(x_shape) in (1, 2)
                if len(x_shape) == 2:
                    x = torch.tensor(x[None].expand(x_shape[0], x_shape[1]))
            self.alpha_raw, self.beta_l_raw, self.beta_s_raw = None, None, None
            if alpha_out or use_beta:
                assert use_split
            if alpha_out is None:
                self.alpha_raw = nn.Parameter(x)
                if use_beta:
                    self.beta_l_raw = nn.Parameter(x)
                    self.beta_s_raw = nn.Parameter(x)
            self.alpha_out = alpha_out
            self.expand_num = out_channels // split
            self.expand_beta_num = in_channels // split

            if auto_lr:
                self.lr_raw = nn.Parameter(torch.tensor([1.]))
            else:
                self.lr_raw = param_multiply

            self.weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_channels, ))
            else:
                self.register_parameter('bias', None)

        if use_bn:
            self.share_bn = ShareBN(out_channels, affine=False)

        nn.init.normal_(self.weight, 0, 0.01)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def gumbel_softmax(self, logits, temperature, topk=1, summary_name=('raw', 'params')):
        """
        input: (..., choice_num)
        return: (..., choice_num), an one-zero vector
        """
        if 'raw' in summary_name:
            logits = logits * self.lr_raw
        else:
            logits = logits * self.param_beta_multiply

        if self.fixed_param is not None:
            logits_softmax = self.fixed_param
        else:
            logits_softmax = logits.softmax(-1)
        if self.training:
            summary = {}
            logits_summary = logits.view(-1)
            for i, l in enumerate(logits_summary):
                summary['dyn_dila_{}_{}'.format(self.count, i)] = l.detach().item()
            add_summary(summary_name[0], **summary)

            summary = {}
            logits_summary = logits_softmax.view(-1)
            for i, l in enumerate(logits_summary):
                summary['dyn_dila_{}_{}'.format(self.count, i)] = l.detach().item()
            add_summary(summary_name[1], **summary)

        if self.end_after_end is False:
            end_search = False
        elif self.end_after_end is True:
            end_search = (get_epoch() > self.tau_end_epoch)
        else:
            end_search = (get_epoch() > self.end_after_end)
        if self.training and not end_search and self.fixed_param is None:
            empty_tensor = logits.new_zeros(logits.size())
            U = nn.init.uniform_(empty_tensor)
            gumbel_sample = -torch.autograd.Variable(torch.log(-torch.log(U + 1e-20) + 1e-20))
            y = F.softmax((logits_softmax.log() + gumbel_sample) / temperature, dim=-1)
        else:
            y = logits
        shape = y.size()
        _, inds = y.topk(topk, dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, inds.view(-1, topk), 1)
        y_hard = y_hard.view(*shape)
        if self.training and not end_search and self.fixed_param is None:
            return ((y_hard - y).detach() + y) / topk
        return y_hard / topk

    def get_tau(self):
        if self.tau_end_epoch is None:
            self.tau_end_epoch = get_total_epoch()
        epoch = get_epoch()
        if self.tau_start_epoch <= epoch <= self.tau_end_epoch:
            t = (self.tau_end_epoch - self.tau_start_epoch + 1) * get_total_inner_iter()
            x = (epoch - self.tau_start_epoch) * get_total_inner_iter() + get_inner_iter()
            if self.sin_tau:
                tau = (self.tau[0] - self.tau[1]) * math.sin(math.pi / 2 / t * x +
                                                             math.pi) + self.tau[0]
            else:
                tau = (self.tau[1] - self.tau[0]) / t * x + self.tau[0]
        else:
            tau = self.tau[1]
        return tau

    def forward_split(self, x):
        # print(self.alpha_raw, self.alpha_raw.grad)
        b_d1, b_d2 = None, None
        x1, x2 = x, x
        if self.alpha_out is not None:
            split = int(self.out_channels * self.alpha_out)
            w_d1, w_d2 = self.weight[split:], self.weight[:split]
            if self.bias is not None:
                b_d1, b_d2 = self.bias[split:], self.bias[:split]
        else:
            tau = self.get_tau()
            lr_summary = self.lr_raw.detach().item() if self.auto_lr else self.lr_raw
            add_summary('tau', tau=tau, avg=self.alpha_raw.abs().mean().detach().item(),
                        lr_raw=lr_summary)
            alpha_tensor = self.gumbel_softmax(self.alpha_raw, temperature=tau).cumsum(-1)
            alpha_tensor = alpha_tensor[:, None].expand(
                alpha_tensor.size(0), self.expand_num).reshape(-1)
            d1_idx = alpha_tensor.bool()
            d2_idx = (1 - alpha_tensor).bool()
            w_d1 = self.weight[d1_idx]
            w_d2 = self.weight[d2_idx]
            if self.beta_s_raw is not None:
                beta_s_tensor = self.gumbel_softmax(
                    self.beta_s_raw, temperature=tau,
                    summary_name=('raw_betas', 'params_betas')).cumsum(-1)
                beta_s_tensor = beta_s_tensor[:, None].expand(
                    beta_s_tensor.size(0), self.expand_beta_num).reshape(-1)
                s_idx = beta_s_tensor.bool()
                w_d1 = w_d1[:, s_idx] * beta_s_tensor[s_idx][None, :, None, None]
                x1 = x1[:, s_idx]

                beta_l_tensor = self.gumbel_softmax(
                    self.beta_l_raw, temperature=tau,
                    summary_name=('raw_betal', 'params_betal')).cumsum(-1)
                beta_l_tensor = beta_l_tensor[:, None].expand(
                    beta_l_tensor.size(0), self.expand_beta_num).reshape(-1)
                l_idx = beta_l_tensor.bool()
                l_idx_rev = l_idx.cpu().numpy()[::-1].copy()
            if self.bias is not None:
                b_d1 = self.bias[d1_idx] * alpha_tensor[d1_idx]
                b_d2 = self.bias[d2_idx] * (1 - alpha_tensor)[d2_idx]

        y = F.conv2d(x1, w_d1, b_d1, self.stride, 1, 1, 1)
        y2 = None
        if self.alpha_out is not None and self.alpha_out > 0 or \
                self.alpha_out is None and (1 - alpha_tensor).sum().detach().item() > 0:
            if self.beta_s_raw is not None:
                w_d2 = w_d2[:, l_idx] * beta_l_tensor[l_idx_rev][None, :, None, None]
                x2 = x2[:, l_idx]
            y2 = F.conv2d(x2, w_d2, b_d2, self.stride, 2, 2, 1)

        if self.use_bn:
            y, y2 = self.share_bn((y, y2))
        y = y * alpha_tensor[d1_idx][:, None, None]
        if y2 is not None:
            y2 = y2 * (1 - alpha_tensor)[d2_idx][:, None, None]
            y = torch.cat([y2, y], dim=1)

        return y

    def forward_select(self, x):
        tau = self.get_tau()
        lr_summary = self.lr_raw.detach().item() if self.auto_lr else self.lr_raw
        add_summary('tau', tau=tau, avg=self.alpha_raw.abs().mean().detach().item(),
                    lr_raw=lr_summary)
        alpha_tensor = self.gumbel_softmax(self.alpha_raw, temperature=tau)

        ys = []
        for block_i in range(self.split):
            start = block_i * self.expand_num
            end = (block_i + 1) * self.expand_num
            bias = self.bias[start:end] if self.bias is not None else None
            for i, dilation in enumerate(self.dilation_choice):
                if alpha_tensor[block_i, i] == 1.:
                    y = F.conv2d(x, self.weight[start:end] * alpha_tensor[block_i, i],
                                 bias, self.stride, dilation, dilation, 1)
                    break

            ys.append(y)
        y = torch.cat(ys, dim=1)
        return y

    def forward(self, x):
        if self.kernel_size == 1 or get_epoch() < self.tau_start_epoch:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                            self.dilation, self.groups)

        if self.use_split:
            return self.forward_split(x)
        return self.forward_select(x)
