import os

o1 = ['heatmap', 'wh_feats', 'reg_offset', 'raw_features', 'heatmap_indexs']
o2 = [
    'heatmap', 'height_feats', 'reg_xoffset', 'reg_yoffset', 'pose',
    'raw_features', 'heatmap_indexs'
]
c1 = ['car', 'bus', 'truck', 'person', 'bicycle', 'tricycle', 'block']
c2 = [
    'right20',
    'right40',
    'right45',
    'left20',
    'left40',
    'left45',
    'NO31',
    'NO32',
    'NO33',
    'NO34',
    'NO35',
    'NO36',
]

model_zoo = {
    'fp16': {
        'config':
        'configs/od/centernet_r34_fp16_nshortcut_giou_raw_ntri_s4_transconv_head12_5x.py',
        'model_file':
        '/private/ningqingqun/torch/centernet/r34_fp16_epoch_16_iter_60000.pth',
        'output': o1,
    },
    'v4.0.0': {
        'config':
        'configs/od/centernet_r18_nshortcut_giou_raw_ntri_s4_transconv_head12_5x.py',
        'model_file':
        '/private/ningqingqun/torch/centernet/r18_epoch_24_iter_56608.pth',
        'output': o1,
    },
    'darknet53': {
        'config': 'configs/od/eft53_o16_v3l_3lr_wd3e4_s123_nos_2x.py',
        'model_file':
        '/private/ningqingqun/torch/centernet/d53_epoch_24_iter_273062.pth',
        'output': o1,
    },
    'v5.1.16': {
        'config': 'configs/fabu3d/centernet_r18_ignore.py',
        'model_file':
        '/private/ningqingqun/mmdet/outputs/v5.1.16/centernet_r18_ignore_1017_1915_gpu12/epoch_35_iter_3675.pth',
        'output': o2,
    },
    'v5.tmp': {
        'config': 'configs/fabu3d/centernet_r18_ignore.py',
        'model_file':
        '/private/ningqingqun/mmdet/outputs/v5.3.2/centernet_r18_ignore_1109_1635_gpu11/epoch_35_iter_8820.pth',
        'output': o2,
    },
    'cm-v0.1': {
        'config': 'configs/crane_mark/centernet_r18_no.py',
        'model_file':
        'work_dirs/debug/centernet_r18_no_1119_1954_desktop/epoch_35_iter_4305.pth',
        'output': o1,
    },
    'cm-v0.2': {
        'config': 'configs/crane_mark/centernet_r18_no.py',
        'model_file':
        'work_dirs/debug/centernet_r18_no_1120_1157_desktop/epoch_40_iter_4920.pth',
        'output': o1,
    },
    'cm-v0.3': {
        'config': 'configs/crane_mark/centernet_r18_no.py',
        'model_file':
        'work_dirs/crane_mark/centernet_r18_no_1120_1458_desktop/epoch_40_iter_4920.pth',
        'output': o1,
    },
    'cm-v0.4': {
        'config': 'configs/crane_mark/centernet_r18_no.py',
        'model_file':
        '/private/ningqingqun/mmdet/outputs/extend_NO/centernet_r18_adam_1122_2207_gpu13/epoch_40_iter_6400.pth',
        'output': o1,
    },
    'cm-0.5': {
        'config': 'configs/crane_mark/centernet_r18_adam.py',
        'model_file':
        '/private/ningqingqun/mmdet/outputs/no31_34/centernet_r18_no_1128_1331_gpu15/epoch_40_iter_6960.pth',
        'output': o1,
    },
    'cm-v0.6': {
        'config': 'configs/crane_mark/centernet_r18_no.py',
        'model_file':
        '/private/ningqingqun/mmdet/outputs/no31_36/centernet_r18_adam_no_crop_1129_1920_gpu9/epoch_10_iter_2000.pth',
        'output': o1,
    },
    'cm-v0.8': {
        'config': 'configs/crane_mark/centernet_r18_no.py',
        'model_file':
        '/private/ningqingqun/mmdet/outputs/cm-v0.8/centernet_r18_finetune_large_1207_1707_desktop/epoch_20_iter_1160.pth',
        'output': o1,
    },
    'cm-v0.9': {
        'config': 'configs/crane_mark/centernet_r18_adam_no_crop.py',
        'model_file':
        '/private/ningqingqun/mmdet/outputs/cm-v0.9/centernet_r18_adam_no_crop_1224_0311_gpu15/epoch_25_iter_14350.pth',
        'output': o1,
    },
    'qr': {
        'config': 'configs/qr/centernet_r18_1x.py',
        'model_file':
        'work_dirs/debug/centernet_r18_1x_1024_2202_desktop/epoch_12_iter_10009.pth',
        'output': o1,
        'classes': ['qr'],
    },
}

model_zoo['v5.5.2'] = dict(
    config=
    'configs/fabu3d/centernet_r34_nshortcut_giou_raw_ntri_s4_transconv_head12_5x.py',
    model_file=
    '/private/ningqingqun/mmdet/outputs/v5.5.2/centernet_r34_nshortcut_giou_raw_ntri_s4_transconv_head12_5x_1220_2050_gpu11/epoch_25_iter_23800.pth',
    output=o2,
    classes=c1)

model_zoo['v5.5.1'] = dict(
    config=
    'configs/fabu3d/centernet_r34_nshortcut_giou_raw_ntri_s4_transconv_head12_5x.py',
    model_file=
    '/private/ningqingqun/mmdet/outputs/v5.5.1/centernet_r34_nshortcut_giou_raw_ntri_s4_transconv_head12_5x_1219_2049_gpu11/epoch_25_iter_19750.pth',
    output=o2,
    classes=c1)

model_zoo['v5.4.2'] = dict(
    config='configs/fabu3d/centernet_r18_ignore.py',
    model_file=
    '/private/ningqingqun/mmdet/outputs/v5.4.2/centernet_r18_ignore_1217_1537_gpu11/epoch_25_iter_16900.pth',
    output=o2,
    classes=c1)

model_zoo['v4.1.1'] = dict(
    config=
    'configs/od/centernet_r34_nshortcut_giou_raw_ntri_s4_transconv_head12_5x.py',
    model_file=
    '/private/ningqingqun/torch/centernet/r34_epoch_24_iter_126179.pth',
    output=o1,
    classes=c1)

model_zoo['v4.2.1'] = dict(
    config=
    'configs/od/centernet_r34_nshortcut_giou_raw_ntri_s4_transconv_head12_5x.py',
    model_file=
    '/private/ningqingqun/mmdet/outputs/v4.2.1/centernet_r34_8cls_merge_1213_1915_gpu12/epoch_20_iter_47120.pth',
    output=o1,
    classes=c1)
