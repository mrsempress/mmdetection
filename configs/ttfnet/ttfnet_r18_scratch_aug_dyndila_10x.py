# model settings
model = dict(
    type='TTFNet',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        local_conv_cfg=dict(type='DynDilaConv', param_multiply=100., alpha_scale=1e-4, split=8,
                            tau_end_epoch=20, tau=(10., 0.5)),
        stage_with_local_conv_cfg=(False, False, True, True),
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_eval=False,
        zero_init_residual=False,
        style='pytorch'),
    neck=None,
    bbox_head=dict(
        type='TTFHead',
        inplanes=(64, 128, 256, 512),
        head_conv=128,
        wh_conv=64,
        hm_head_conv_num=2,
        wh_head_conv_num=1,
        num_classes=81,
        wh_offset_base=16,
        wh_agnostic=True,
        wh_gaussian=True,
        shortcut_cfg=(1, 2, 3),
        norm_cfg=dict(type='BN'),
        alpha=0.54,
        hm_weight=1.,
        wh_weight=5.))
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    debug=False)
test_cfg = dict(
    score_thr=0.01,
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0004,
                 paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.,
                                        arch_lr_mult=1e3, arch_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 5,
    step=[90, 110])
checkpoint_config = dict(save_every_n_steps=200, max_to_keep=1, keep_in_n_epoch=[90])
bbox_head_hist_config = dict(
    model_type=['ConvModule', 'DeformConvPack'],
    sub_modules=['bbox_head'],
    save_every_n_steps=200)
# yapf:disable
log_config = dict(interval=20)
# yapf:enable
# runtime settings
total_epochs = 120
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ttfnet18_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
