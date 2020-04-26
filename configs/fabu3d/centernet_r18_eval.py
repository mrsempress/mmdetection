model = dict(
    type='CenterNet2',
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
dataset_type = 'Vision2DDataset'
data_root = '/private/ningqingqun/obstacle_detector/data/obstacle2d/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=20,
    workers_per_gpu=4,
    test=dict(
        type=dataset_type,
        task='8cls_merge',
        ann_file=data_root + 'ImageSets/val1010.txt',
        img_prefix=data_root + 'JPGImages/',
        img_scale=(1280, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True,
        keep_difficult=False,
        resize_keep_ratio=False))
# yapf:disable
log_config = dict(interval=100)
# yapf:enable
# runtime settings
total_epochs = 120
device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'centernet_r18_nshortcut_giou_raw_ntri_s4_transconv_head12_5x'
load_from = '/private/ningqingqun/torch/centernet/r18_epoch_24_iter_56608.pth'
# load_from = None
resume_from = None
workflow = [('train', 1)]