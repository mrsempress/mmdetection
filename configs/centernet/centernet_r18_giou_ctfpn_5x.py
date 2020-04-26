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
    neck=dict(
        type='CTFPN',
        inplanes=[64, 128, 256, 512],
        planes=[256, 128, 64],
        outplane=64,
        norm_cfg=None,
        norm_eval=False),
    bbox_head=dict(
        type='MLCTHead',
        inplane=64,
        fpn_strides=(4, 8, 16),
        head_conv=128,
        stacked_convs=1,
        obj_sizes_of_interest=((-1, 128), (96, 384), (256, 99999)),
        num_classes=81,
        use_smooth_l1=True,
        use_giou=True,
        use_nms=True,
        use_exp_wh=False,
        conv_cfg=None,
        norm_eval=False,
        select_feat_index=None,
        wh_weight=0.1,
        hm_weight=1.,
        gamma=2.))
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    debug=False)
test_cfg = dict(
    score_thr=0.01,
    max_per_level=64,
    max_per_img=100,
    nms=dict(type='nms', iou_thr=0.5))
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
            dict(type='Normalize', **img_norm_cfg),
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
optimizer = dict(type='Adam', lr=7.02e-5)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[40, 55])
checkpoint_config = dict(save_every_n_steps=200, max_to_keep=1, keep_every_n_epochs=40)
backbone_hist_config = dict(
    model_type=['BasicBlock'],
    sub_modules=['backbone'],
    save_every_n_steps=200)
neck_hist_config = dict(
    model_type=['Conv2d', 'ModulatedDeformConvPack'],
    sub_modules=['neck'],
    save_every_n_steps=200)
bbox_head_hist_config = dict(
    model_type=['ConvModule'],
    sub_modules=['bbox_head'],
    save_every_n_steps=200)
# yapf:disable
log_config = dict(interval=100)
# yapf:enable
# runtime settings
total_epochs = 60
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'centernet_r18_giou_ctfpn_5x'
load_from = None
resume_from = None
workflow = [('train', 1)]
