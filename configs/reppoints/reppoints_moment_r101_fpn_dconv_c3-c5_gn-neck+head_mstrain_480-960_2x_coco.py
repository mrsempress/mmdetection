_base_ = './reppoints_moment_r50_fpn_gn-neck+head_mstrain_480-960_2x_coco.py'
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(
        depth=101,
        dcn=dict(type='DCN', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
work_dir = './work_dirs/reppoints_moment_r101_dcn_fpn_2x_mt'