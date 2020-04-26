import torch
import torch.nn as nn

from mmdet.ops import ModulatedDeformConvPack

OPS = {
    'none': lambda inc, outc, stride, upsample, affine: Zero(inc, outc),
    'skip_connect': lambda inc, outc, stride, upsample, affine: Identity(
        inc, outc, upsample, affine=affine) if stride == 1 else \
        FactorizedReduce(inc, outc, affine=affine),
    'conv_1x1': lambda inc, outc, stride, upsample, affine: ReLUConvBN(
        inc, outc, 1, stride, 0, upsample, affine),
    'conv_3x3': lambda inc, outc, stride, upsample, affine: ReLUConvBN(
        inc, outc, 3, stride, 1, upsample, affine),
    'sep_conv_3x3': lambda inc, outc, stride, upsample, affine: SepConv(
        inc, outc, 3, stride, 1, upsample, affine),
    'sep_conv_5x5': lambda inc, outc, stride, upsample, affine: SepConv(
        inc, outc, 5, stride, 2, upsample, affine),
    'sep_conv_7x7': lambda inc, outc, stride, upsample, affine: SepConv(
        inc, outc, 7, stride, 3, upsample, affine),
    'dil_conv_3x3': lambda inc, outc, stride, upsample, affine: DilConv(
        inc, outc, 3, stride, 2, 2, upsample, affine),
    'dil_conv_5x5': lambda inc, outc, stride, upsample, affine: DilConv(
        inc, outc, 5, stride, 4, 2, upsample, affine),
    'mdcn_3x3': lambda inc, outc, stride, upsample, affine: MDConv(
        inc, outc, 3, stride, 1, upsample, affine)
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, upsample, affine=True):
        super(ReLUConvBN, self).__init__()
        op = [nn.ReLU(inplace=False),
              nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=upsample)]
        if upsample:
            op.append(nn.UpsamplingBilinear2d(scale_factor=2))
        op.append(nn.BatchNorm2d(C_out, affine=affine))
        self.op = nn.Sequential(*op)

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, upsample, affine=True):
        super(DilConv, self).__init__()
        op = [nn.ReLU(inplace=False),
              nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                        dilation=dilation, groups=C_in, bias=False),
              nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=upsample)]
        if upsample:
            op.append(nn.UpsamplingBilinear2d(scale_factor=2))
        op.append(nn.BatchNorm2d(C_out, affine=affine))
        self.op = nn.Sequential(*op)

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, upsample, affine=True):
        super(SepConv, self).__init__()
        op = [nn.ReLU(inplace=False),
              nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                        groups=C_in, bias=False),
              nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
              nn.BatchNorm2d(C_in, affine=affine),
              nn.ReLU(inplace=False),
              nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding,
                        groups=C_in,
                        bias=False),
              nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=upsample)]
        if upsample:
            op.append(nn.UpsamplingBilinear2d(scale_factor=2))
        op.append(nn.BatchNorm2d(C_out, affine=affine))
        self.op = nn.Sequential(*op)

    def forward(self, x):
        return self.op(x)


class MDConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, upsample, affine=True):
        super(MDConv, self).__init__()
        op = [nn.ReLU(inplace=False),
              ModulatedDeformConvPack(C_in, C_out, kernel_size, stride, padding, bias=upsample)]
        if upsample:
            op.append(nn.UpsamplingBilinear2d(scale_factor=2))
        op.append(nn.BatchNorm2d(C_out, affine=affine))
        self.op = nn.Sequential(*op)

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self, C_in, C_out, upsample, affine=True):
        super(Identity, self).__init__()
        assert (upsample and C_in != C_out) or (not upsample and C_in == C_out)
        if C_in != C_out:
            self.conv_1x1 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_out, 1, bias=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        if hasattr(self, 'conv_1x1'):
            return self.conv_1x1(x)
        return x


class Zero(nn.Module):

    def __init__(self, inc, outc):
        super(Zero, self).__init__()
        self.inc = inc
        self.outc = outc

    def forward(self, x):
        N, C, H, W = x.shape
        if self.inc < self.outc:
            return x.new_zeros((N, self.outc, H // 2, W // 2))
        elif self.inc > self.outc:
            return x.new_zeros((N, self.outc, H * 2, W * 2))
        return x.new_zeros((N, self.outc, H, W))


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
