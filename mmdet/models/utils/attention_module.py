import torch
import torch.nn as nn

from mmdet.ops import DeformConvPack
from .norm import build_norm_layer


class SEBlock(nn.Module):

    def __init__(self,
                 planes,
                 num_levels=1,
                 num_scales=1,
                 compress_ratio=16):
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.ModuleList(
            [nn.Conv2d(planes * num_levels, planes * num_levels // compress_ratio,
                       1, stride=1, padding=0)] * num_scales)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.ModuleList(
            [nn.Conv2d(planes * num_levels // compress_ratio, planes * num_levels,
                       1, stride=1, padding=0)] * num_scales)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """

        Args:
            x: tensor or list(tensor).

        Returns:
            tensor or list(tensor)
        """
        attention_feats = []
        feats = x if isinstance(x, (tuple, list)) else [x]
        for i, mf in enumerate(feats):
            avg = self.avgpool(mf)
            fc1 = self.relu(self.fc1[i](avg))
            fc2 = self.sigmoid(self.fc2[i](fc1))
            attention_feats.append(mf * fc2)
        attention_feats = attention_feats if isinstance(x, (tuple, list)) \
            else attention_feats[0]
        return attention_feats


class CBAMBlock(nn.Module):

    def __init__(self,
                 use_channel_attention=True,
                 use_spatial_attention=True,
                 planes=None,
                 compress_ratio=16,
                 kernel_size=7,
                 padding=3):
        super(CBAMBlock, self).__init__()
        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention

        if use_channel_attention:
            assert planes is not None
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)

            self.fc1 = nn.Conv2d(planes, planes // compress_ratio, 1)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Conv2d(planes // compress_ratio, planes, 1)
            self.sigmoid_c = nn.Sigmoid()

        if use_spatial_attention:
            self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding)
            self.sigmoid_s = nn.Sigmoid()

    def forward(self, x):
        if self.use_channel_attention:
            avg_out_c = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
            max_out_c = self.fc2(self.relu(self.fc1(self.max_pool(x))))
            channel_attention = self.sigmoid_c(avg_out_c + max_out_c)
            x = x * channel_attention

        if self.use_spatial_attention:
            avg_out_s = torch.mean(x, dim=1, keepdim=True)
            max_out_s, _ = torch.max(x, dim=1, keepdim=True)
            raw_attention_map = self.conv1(torch.cat([avg_out_s, max_out_s], dim=1))
            spatial_attention = self.sigmoid_s(raw_attention_map)
            x = x * spatial_attention
        return x


class MaskModule(nn.Module):

    def __init__(self,
                 inplain,
                 feat_idx=None,
                 feat_stride=4,
                 plains=(128, 64, 32, 1),
                 norm_cfg=dict(type='BN'),
                 pos_thre=0.1):
        super(MaskModule, self).__init__()
        self.feat_idx = feat_idx
        self.feat_stride = feat_stride
        self.feat_base_size = 32 // feat_stride  # 32 is size_divisor
        self.pos_thre = pos_thre

        layers = []
        for i in range(len(plains)):
            inp = inplain if i == 0 else plains[i - 1]
            layers.append(DeformConvPack(inp, plains[i], 3, padding=1))
            if norm_cfg:
                layers.append(build_norm_layer(norm_cfg, plains[i])[1])
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x[self.feat_idx] if self.feat_idx else x
        mask = self.sigmoid(self.layers(x))  # N, 1, H, W

        N, C, H, W = mask.shape
        mask_corner = []
        for img_id in range(N):
            pos_ind = (mask[img_id, 0, :] > self.pos_thre).nonzero()
            (x1, y1), (x2, y2) = pos_ind.min(0)[0], pos_ind.max(0)[0] + 1  # all int
            x1 = (x1 // self.feat_base_size) * self.feat_base_size
            y1 = (y1 // self.feat_base_size) * self.feat_base_size
            x2 = ((x2 // self.feat_base_size + 1) * self.feat_base_size).clamp(max=W)
            y2 = ((y2 // self.feat_base_size + 1) * self.feat_base_size).clamp(max=H)
            # print(x1.requires_grad)
            mask_corner.append([x1, y1, x2, y2])

        if not self.training:
            assert N == 1, NotImplementedError
            x1, y1, x2, y2 = mask_corner[0]
            x = x[:, :, y1:y2, x1:x2]

        mask_corner = x.new_tensor(mask_corner)
        # print(mask_corner.requires_grad)

        return x, (mask, mask_corner, self.feat_stride)
