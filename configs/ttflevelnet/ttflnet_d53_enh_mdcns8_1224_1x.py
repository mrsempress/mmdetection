# model settings
model = dict(
    type='TTFNet',
    pretrained='./pretrain/darknet53.pth',
    backbone=dict(
        type='DarknetV3',
        layers=[1, 2, 8, 8, 4],
        inplanes=[3, 32, 64, 128, 256, 512],
        planes=[32, 64, 128, 256, 512, 1024],
        norm_cfg=dict(type='BN'),
        out_indices=(1, 2, 3, 4),
        frozen_stages=1,
        norm_eval=False),
    neck=None,
    bbox_head=dict(
        type='TTFLevelHead',
        inplanes=(128, 256, 512, 1024),
        planes=(256, 128, 64),
        down_ratio_b1=8,
        down_ratio_b2=4,
        hm_head_channels=(128, 128),
        wh_head_channels=(64, 64),
        hm_head_conv_num=(2, 2),
        wh_head_conv_num=(2, 2),
        num_classes=81,
        wh_scale_factor_b1=8.,
        wh_scale_factor_b2=8.,
        shortcut_cfg=(1, 2, 3),
        extra_shortcut_cfg=3,
        alpha=0.6,
        beta=0.6,
        max_objs=128,
        hm_weight_b1=1.,
        wh_weight_b1=5.,
        hm_weight_b2=1.,
        wh_weight_b2=5.,
        b1_min_length=64,
        b2_max_length=64,
        mdcn_before_s8=True,
        inf_branch=['b1', 'b2'],
        use_simple_nms=True,
        conv_cfg=None,
        norm_cfg=dict(type='BN')))
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(debug=False)
test_cfg = dict(score_thr=0.01, max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
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
    imgs_per_gpu=12,
    workers_per_gpu=2,
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
    lr=0.002,
    momentum=0.9,
    weight_decay=0.0004,
    paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    step=[9, 11])
checkpoint_config = dict(save_every_n_steps=200, max_to_keep=1, keep_every_n_epochs=9)
# yapf:disable
log_config = dict(interval=20)
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ttfv2net_d53_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
