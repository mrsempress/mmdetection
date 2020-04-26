# model settings
model = dict(
    type='TTFNeXt',
    pretrained='modelzoo://resnet18',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_eval=False,
        style='pytorch'),
    neck=None,
    bbox_head=dict(
        type='TTFXHead',
        search_k=4,
        search_op_gumbel_softmax=True,
        search_edge_gumbel_softmax=True,
        gumbel_no_affine=True,
        inplanes=(64, 128, 256, 512),
        planes=(64, 128, 256, 512),
        ops_list=('skip_connect', 'conv_1x1', 'conv_3x3', 'sep_conv_3x3',
                  'sep_conv_5x5', 'sep_conv_7x7', 'dil_conv_3x3', 'dil_conv_5x5',
                  'mdcn_3x3'),
        head_conv=128,
        wh_conv=64,
        hm_head_conv_num=2,
        wh_head_conv_num=1,
        num_classes=81,
        wh_offset_base=16,
        wh_gaussian=True,
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
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    search=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
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
                 paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
search_config = dict(
    tune_epoch_start=4,
    tune_epoch_end=8,
    search_optimizer=dict(
        type='Adam',
        lr=5e-4,
        betas=(0.5, 0.999),
        weight_decay=1e-3))
# learning policy
lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 5)
checkpoint_config = dict(save_every_n_steps=200, max_to_keep=1, keep_in_n_epoch=[4])
bbox_head_hist_config = dict(
    model_type=['ConvModule', 'DeformConvPack'],
    sub_modules=['bbox_head'],
    save_every_n_steps=200)
# yapf:disable
log_config = dict(interval=20)
# yapf:enable
# runtime settings
total_epochs = 8
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'ttfnext18_search_gumbel_non_wd5e4_1x'
load_from = None
resume_from = 'work_dirs/1909/0925r_S18_8EPOCH_GUMBEL_NOAFFINE_5E4_WD3E4_NONE_NOSEED_1X/ttfnext18_search_gumbel_nons_wd3e4_lr5e4_1x_0925_2234/epoch_4.pth'
workflow = [('train', 1)]
