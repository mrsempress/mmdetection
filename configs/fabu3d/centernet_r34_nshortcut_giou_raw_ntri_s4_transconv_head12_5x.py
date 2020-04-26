# model settings
model = dict(
    type='CenterNet2',
    pretrained='modelzoo://resnet34',
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_eval=False,
        add_summay_every_n_step=200,
        style='pytorch'),
    neck=dict(type='None'),
    bbox_head=dict(
        type='CTBiHead',
        inplanes=(64, 128, 256, 512),
        planes=(256, 128, 64),
        head_conv=64,
        hm_head_conv_num=2,
        wh_head_conv_num=1,
        deconv_with_bias=False,
        num_classes=8,
        use_smooth_l1=False,
        use_shortcut=True,
        shortcut_in_shortcut=True,
        shortcut_cfg=(1, 2, 3),
        use_trident=True,
        hm_init_value=-6.,
        norm_cfg=dict(type='BN'),
        heights_weight=0.5,
        hm_weight=1.))
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(vis_every_n_iters=500, debug=False)
test_cfg = dict(score_thr=0.01, max_per_img=100)
# dataset settings
dataset_type = 'Vision3DDataset'
data_root = 'data/fabu3d/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'pkl/train_merge.pkl',
        img_prefix=data_root + 'images/',
        img_scale=(1280, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_ignore=True,
        extra_aug=dict(
            photo_metric_distortion=dict(
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            expand=dict(
                mean=img_norm_cfg['mean'],
                to_rgb=img_norm_cfg['to_rgb'],
                ratio_range=(1, 3)),
            random_corner_crop=dict(min_crop_sizes=(0.3, 0.5, 0.7, 0.9))),
        resize_keep_ratio=False),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'pkl/val_94ff72bd.pkl',
        img_prefix=data_root + 'images/',
        img_scale=(1280, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_ignore=True,
        resize_keep_ratio=False),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'pkl/val_94ff72bd.pkl',
        img_prefix=data_root + 'images/',
        img_scale=(1280, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_ignore=True,
        resize_keep_ratio=False))
# optimizer
optimizer = dict(type='Adam', lr=7.02e-5)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[20])
checkpoint_config = dict(
    save_every_n_steps=500, max_to_keep=1, keep_every_n_epochs=16)
bbox_head_hist_config = dict(
    model_type=['ConvModule', 'TridentConv2d'],
    sub_modules=['bbox_head'],
    save_every_n_steps=500)
# yapf:disable
log_config = dict(interval=100)
evaluation = dict(interval=1, cache_result=True)
# yapf:enable
# runtime settings
total_epochs = 25
device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'centernet_r34_nshortcut_giou_raw_ntri_s4_transconv_head12_5x'
load_from = '/private/ningqingqun/mmdet/outputs/v4.2.1/centernet_r34_8cls_merge_1213_1915_gpu12/epoch_20_iter_47120.pth'
#load_from = None
resume_from = None
workflow = [('train', 1)]
