# model settings
model = dict(
    type='MoCo',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        return_fc=True,
        num_classes=128,
        norm_eval=False,
        zero_init_residual=False,
        style='pytorch'),
    len_queue=65536,
    momentum=0.999,
    temperature=0.2,
    mlp=True)
train_cfg = dict(vis_freq=100)
test_cfg = None
# dataset settings
dataset_type = 'TwoCropsDataset'
data_root = '/data/liuzili/data/public_datasets/Cityscapes/leftImg8bit/trainval/'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
mocov1_pipeline = [
    dict(type='ImgRandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='ImgRandomGrayscale', p=0.2),
    dict(type='ImgColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    dict(type='ImgRandomHorizontalFlip', p=0.5),
    dict(type='ImgToTensor'),
    dict(type='ImgNormalize', **img_norm_cfg),
]
mocov2_pipeline = [
    dict(type='ImgRandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='ImgRandomApplyColorJitter',
         p=0.8, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    dict(type='ImgRandomGrayscale', p=0.2),
    dict(type='ImgRandomApplyGaussianBlur', p=0.5, sigma=[.1, 2.]),
    dict(type='ImgRandomHorizontalFlip', p=0.5),
    dict(type='ImgToTensor'),
    dict(type='ImgNormalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=64,
    workers_per_gpu=32,
    train=dict(
        type=dataset_type,
        img_prefix=data_root + 'train/',
        pipeline=mocov2_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.001875,  # lr when using 16-GPUs
    momentum=0.9,
    weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[120, 160])
checkpoint_config = dict(save_every_n_steps=2000, max_to_keep=1, keep_every_n_epochs=120)
# yapf:disable
log_config = dict(interval=10)
# yapf:enable
# runtime settings
total_epochs = 200
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'r50_moco'
load_from = None
resume_from = None
workflow = [('train', 1)]
