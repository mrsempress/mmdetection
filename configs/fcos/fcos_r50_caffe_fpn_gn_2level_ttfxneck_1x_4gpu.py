# model settings
model = dict(
    type='FCOS',
    pretrained='open-mmlab://resnet50_caffe',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        add_summay_every_n_step=500,
        style='caffe'),
    neck=dict(
        type='TTFXFPN',
        inplanes=(256, 512, 1024, 2048),
        planes=(128, 128, 128),
        shortcut_cfg=(1, 2, 3),
        use_extra_shortcut=True,
        s16_shortcut_twice=True,
        shortcut_conv_cfg=None,
        up_conv_cfg=None,
        upsample_vallina=False,
        dcn_offset_mean=False,
        down_ratio=(16, 8),
        norm_cfg=None),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=81,
        in_channels=128,
        stacked_convs=4,
        feat_channels=128,
        simple_nms=False,
        strides=[8, 16],
        regress_ranges=((-1, 256), (256, 1e8)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=1,
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
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(save_every_n_steps=2000, max_to_keep=1, keep_every_n_epochs=100)
neck_hist_config = dict(
    model_type=['Conv2d'],
    sub_modules=['neck'],
    save_every_n_steps=500)
bbox_head_hist_config = dict(
    model_type=['GN', 'Conv2d'],
    sub_modules=['bbox_head'],
    save_every_n_steps=500)
# yapf:disable
log_config = dict(interval=20)
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/fcos_r50_caffe_fpn_gn_1x_4gpu'
load_from = None
resume_from = None
workflow = [('train', 1)]
