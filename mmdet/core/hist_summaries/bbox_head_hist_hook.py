from mmdet.core.utils import HistogramHook


class BBoxHeadHistHook(HistogramHook):

    def __init__(self, model_types, save_every_n_steps, sub_modules=('bbox_head',)):
        super(BBoxHeadHistHook, self).__init__('bbox_head_hist',
                                               model_types=model_types,
                                               save_every_n_steps=save_every_n_steps,
                                               sub_modules=sub_modules)
