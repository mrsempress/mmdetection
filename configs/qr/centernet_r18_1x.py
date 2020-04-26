# model settings
model = dict(
    type='CenterNet',
    pretrained='modelzoo://resnet18',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_eval=False,
        add_summay_every_n_step=200,
        style='pytorch'),
    neck=dict(type='None'),
    bbox_head=dict(
        type='CTHead',
        inplanes=(64, 128, 256, 512),
        planes=(256, 128, 128),
        head_conv=128,
        deconv_with_bias=False,
        num_classes=2,
        use_reg_offset=True,
        use_smooth_l1=False,
        use_giou=True,
        use_shortcut=True,
        use_rep_points=False,
        norm_cfg=dict(type='BN'),
        wh_weight=0.1,
        off_weight=1.,
        hm_weight=1.))
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(vis_every_n_iters=500, debug=False)
test_cfg = dict(score_thr=0.1, max_per_img=100)
# dataset settings
dataset_type = 'QrCodeDataset'
data_root = '/private/ningqingqun/datasets/qr_dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.pkl',
        img_prefix=data_root + 'images/',
        img_scale=(1280, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        extra_aug=dict(
            photo_metric_distortion=dict(
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            expand=dict(
                mean=img_norm_cfg['mean'],
                to_rgb=img_norm_cfg['to_rgb'],
                ratio_range=(1, 4)),
            random_crop=dict(
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
        resize_keep_ratio=False),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.pkl',
        img_prefix=data_root + 'images/',
        img_scale=(1280, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        resize_keep_ratio=False),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val.pkl',
        img_prefix=data_root + 'images/',
        img_scale=(1280, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True,
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
    step=[8, 10])
checkpoint_config = dict(
    save_every_n_steps=500, max_to_keep=1, keep_every_n_epochs=16)
# yapf:disable
log_config = dict(interval=100)
# yapf:enable
# runtime settings
total_epochs = 12
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'centernet_r18_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
