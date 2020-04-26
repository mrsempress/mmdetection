from mmdet.core.utils import HistogramHook


class NeckHistHook(HistogramHook):

    def __init__(self, model_types, save_every_n_steps, sub_modules=('fpn',)):
        super(NeckHistHook, self).__init__('fpn_hist',
                                           model_types=model_types,
                                           save_every_n_steps=save_every_n_steps,
                                           sub_modules=sub_modules)

