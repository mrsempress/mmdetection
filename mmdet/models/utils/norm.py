import torch.nn as nn
import torch.nn.functional as F


class ShareBN(nn.BatchNorm2d):

    def __init__(self, *args, **kwargs):
        super(ShareBN, self).__init__(*args, **kwargs)

    def forward(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)
        self._check_input_dim(x_h)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        outc_h = x_h.size(1)
        running_mean, running_var = self.running_mean, self.running_var
        weight, bias = self.weight, self.bias

        if self.track_running_stats:
            running_mean = self.running_mean[-outc_h:]
            running_var = self.running_var[-outc_h:]

        if self.affine:
            weight = self.weight[-outc_h:]
            bias = self.bias[-outc_h:]

        x_h = F.batch_norm(
            x_h, running_mean, running_var, weight, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        if x_l is None:
            return x_h, None

        if self.track_running_stats:
            running_mean = self.running_mean[:-outc_h]
            running_var = self.running_var[:-outc_h]

        if self.affine:
            weight = self.weight[:-outc_h]
            bias = self.bias[:-outc_h]

        x_l = F.batch_norm(
            x_l, running_mean, running_var, weight, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        return x_h, x_l


norm_cfg = {
    # format: layer_type: (abbreviation, module)
    'BN': ('bn', nn.BatchNorm2d),
    'SyncBN': ('bn', nn.SyncBatchNorm),
    'GN': ('gn', nn.GroupNorm),
    'ShareBN': ('bn', ShareBN)
    # and potentially 'SN'
}


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        # can froze mean, std and affine parameters.
        param.requires_grad = requires_grad

    return name, layer
