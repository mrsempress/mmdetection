_base_ = './fovea_r50_fpn_4x4_1x_coco.py'
model = dict(
    bbox_head=dict(
        with_deform=True,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)))
# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 24
work_dir = './work_dirs/fovea_align_gn_r50_fpn_4gpu_2x'