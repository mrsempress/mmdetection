import numpy as np
from tensorboardX import SummaryWriter
from mmcv.runner import master_only

from .viz import draw_boxes


_summary_dict = dict()
_writer = None
_global_step = 0
_local_step = 0
_epoch = 0
_total_epoch = 0
_inner_iter = 0
_total_inner_iter = 1
_default_summary_freq = 1000


@master_only
def add_summary(prefix, average_factor=1, **kwargs):
    for var, value in kwargs.items():
        if prefix:
            _summary_dict['{}/{}'.format(prefix, var)] = value / average_factor
        else:
            _summary_dict['{}'.format(var)] = value / average_factor


def get_summary():
    return _summary_dict


@master_only
def set_writer(dir_name='./tmp'):
    global _writer
    _writer = SummaryWriter(dir_name)
    return _writer


def get_writer():
    global _writer
    assert _writer, "writer is not init."
    return _writer


@master_only
def add_image_summary(tag, image, boxes=None, labels=None, type='0to255', **kwargs):
    global _writer
    assert type in ['0to255', '-101', 'mean0var', '0to1']
    try:
        if type == '0to255' :
            im = image
            if boxes and labels:
                im = im[:, :, [2, 1, 0]]
                labels = list(labels)
                im = draw_boxes(im, boxes, labels)
                im = np.transpose(im, (2, 0, 1))
        elif type == '-101':
            im = image.cpu().numpy()
            im[im == -1] = 128
            im[im == 0] = 0
            im[im == 1] == 255
        elif type == 'mean0var':
            im = image.cpu().numpy()
            im *= np.array((0.229, 0.224, 0.225)).reshape(3, 1, 1)
            im += np.array((0.485, 0.456, 0.406)).reshape(3, 1, 1)
            im = np.array(im * 255., dtype=np.uint8)
        elif type == '0to1':
            im = image.cpu().numpy()
            im = np.array(im * 255., dtype=np.uint8)
        _writer.add_image(tag, im, global_step=_global_step, **kwargs)
    except:
        pass


@master_only
def add_feature_summary(tag, feats, type='all', **kwargs):
    global _writer
    assert type in ['f', 'l', 'mean', 'max', 'all']

    def sigmoid_and_to_255(array):
        return np.round(1.0 / (1 + np.exp(-1 * array)) * 255)

    def add_image_shortcut(tag, feat):
        vis_feat = sigmoid_and_to_255(feat)
        _writer.add_image(tag, vis_feat, global_step=_global_step, **kwargs)

    try:
        if type in ['f', 'all']:
            add_image_shortcut('{}_f'.format(tag), feats[0, [0]])
        if type in ['l', 'all']:
            add_image_shortcut('{}_l'.format(tag), feats[0, [-1]])
        if type in ['mean', 'all']:
            add_image_shortcut('{}_mean'.format(tag), feats[0].mean(0)[None])
        if type in ['max', 'all']:
            add_image_shortcut('{}_max'.format(tag), feats[0].max(0)[None])
    except Exception as e:
        print(e)


@master_only
def add_histogram_summary(tag, tensor, is_param=False, collect_type='all', **kwargs):
    global _writer
    assert collect_type in ['mean', 'max', 'none', 'all']

    def add_histogram_shortcut(tag, tensor):
        _writer.add_histogram(tag, tensor, global_step=_global_step, walltime=3600, **kwargs)

    try:
        if collect_type in ['mean', 'all']:
            if is_param:
                summary_tensor = tensor.reshape((tensor.shape[0], -1)).mean(1)
            else:
                summary_tensor = tensor[0].mean(1).mean(1)
            add_histogram_shortcut('{}/mean'.format(tag), summary_tensor)

        if collect_type in ['max', 'all']:
            if is_param:
                summary_tensor = tensor.reshape((tensor.shape[0], -1)).max(1)[0]
            else:
                summary_tensor = tensor[0].max(1)[0].max(1)[0]
            add_histogram_shortcut('{}/max'.format(tag), summary_tensor)

        if collect_type in ['none', 'all']:
            add_histogram_shortcut(tag, tensor)
    except Exception as e:
        print(e)


def reset_global_step(global_step=0, logger=None):
    global _global_step
    _global_step = global_step
    if logger:
        logger.info("Global step has been set to {}".format(global_step))


def update_global_step(delta_step=1):
    global _global_step
    global _local_step
    _global_step += delta_step
    _local_step += 1


@master_only
def write_txt(data, filename='tmp', thre=None):
    if thre:
        data = data[data > thre]
    data = list(data.view(-1).cpu().numpy())
    with open("./{}.txt".format(filename), 'a+') as f:
        print(data, file=f)


def set_epoch(epoch):
    global _epoch
    _epoch = epoch


def set_total_epoch(total_epoch):
    global _total_epoch
    _total_epoch = total_epoch


def set_inner_iter(inner_iter):
    global _inner_iter
    _inner_iter = inner_iter


def set_total_inner_iter(total_inner_iter):
    global _total_inner_iter
    _total_inner_iter= total_inner_iter


def get_global_step():
    return _global_step


def get_local_step():
    return _local_step


def get_epoch():
    return _epoch


def get_total_epoch():
    return _total_epoch


def get_inner_iter():
    return _inner_iter


def get_total_inner_iter():
    return _total_inner_iter


@master_only
def every_n_local_step(n=_default_summary_freq):
    if _local_step > 0 and _local_step % n == 0:
        return True
    elif _local_step == 100:
        return True
    return False
