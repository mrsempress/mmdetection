from mmdet.core.utils import HistogramHook


class BackboneHistHook(HistogramHook):

    def __init__(self, model_types, save_every_n_steps, sub_modules=('backbone',)):
        super(BackboneHistHook, self).__init__('backbone_hist',
                                               model_types=model_types,
                                               save_every_n_steps=save_every_n_steps,
                                               sub_modules=sub_modules)
