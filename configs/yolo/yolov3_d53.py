# model settings
model = dict(
    type='YOLOv3',
    pretrained='./pretrain/darknet53.pth',
    backbone=dict(
        type='DarknetV3',
        layers=[1, 2, 8, 8, 4],
        inplanes=[3, 32, 64, 128, 256, 512],
        planes=[32, 64, 128, 256, 512, 1024],
        norm_cfg=dict(type='BN'),
        out_indices=(2, 3, 4),
        frozen_stages=1,
        norm_eval=True),
    neck=dict(
        type='DarknetFPN',
        in_channels=(256, 512, 1024),
        out_channels=(256, 512, 1024),
        norm_cfg=dict(type='BN'),
        norm_eval=True),
    bbox_head=dict(
        type='YOLOv3Head',
        in_channels=(256, 512, 1024),
        anchors=((10, 13, 16, 30, 33, 23),
                 (30, 61, 62, 45, 59, 119),
                 (116, 90, 156, 198, 373, 326)),
        strides=(8, 16, 32),
        num_classes=81,
        alloc_size=(128, 128),
        ignore_iou_thresh=0.7))
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(
    debug=False)
test_cfg = dict(
    nms_pre=400,
    min_bbox_size=0,
    score_thr=0.01,
    nms=dict(type='nms', iou_thr=0.45),
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
    dict(type='Resize', img_scale=(608, 608), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
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
optimizer = dict(type='SGD', lr=0.00025, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[2, 4])
checkpoint_config = dict(save_every_n_steps=2500, max_to_keep=1)
# yapf:disable
log_config = dict(interval=100)
# yapf:enable
# runtime settings
total_epochs = 5
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'yolov3_d53_0.5x'
load_from = './pretrain/yolov3_new.pth'
resume_from = None
workflow = [('train', 1)]
