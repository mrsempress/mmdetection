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
        head_conv=64,
        deconv_with_bias=False,
        num_classes=8,
        use_reg_offset=True,
        use_smooth_l1=False,
        use_giou=False,
        use_shortcut=False,
        use_rep_points=False,
        norm_cfg=dict(type='BN'),
        wh_weight=0.1,
        off_weight=1.,
        hm_weight=1.))
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    debug=False)
test_cfg = dict(
    score_thr=0.1,
    max_per_img=100)
# dataset settings
dataset_type = 'Vision2DDataset'
data_root = '/private/ningqingqun/obstacle_detector/data/obstacle2d/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1280, 800), keep_ratio=False),
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
        img_scale=(1280, 800),
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
        ann_file=data_root + 'ImageSets/train190302_nobg.txt',
        pipeline=train_pipeline,
        img_prefix=data_root + 'JPGImages/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/val1010.txt',
        pipeline=test_pipeline,
        img_prefix=data_root + 'JPGImages_update/',),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/val1010.txt',
        pipeline=test_pipeline,
        img_prefix=data_root + 'JPGImages_update/'))
# optimizer
optimizer = dict(type='Adam', lr=7.02e-5)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
checkpoint_config = dict(save_every_n_steps=200, max_to_keep=1, keep_every_n_epochs=16)
# yapf:disable
log_config = dict(interval=100)
# yapf:enable
# runtime settings
total_epochs = 24
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'centernet_r18_1280x800_2x'
load_from = None
resume_from = None
workflow = [('train', 1)]
