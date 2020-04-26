from .single_stage import SingleStageDetector
from .search_base import SearchBaseDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class TTFNeXt(SingleStageDetector, SearchBaseDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TTFNeXt, self).__init__(backbone, neck, bbox_head, train_cfg,
                                      test_cfg, pretrained)

    def arch_parameters(self):
        return self.bbox_head.arch_parameters()

    def genotype(self):
        return self.bbox_head.genotype()

    def plot(self, g, filename):
        return self.bbox_head.plot(g, filename)

    def rebuild(self, g):
        return self.bbox_head.rebuild(g)

    def reset_tau(self, tau):
        return self.bbox_head.reset_tau(tau)

    def reset_do_search(self, do_search):
        return self.bbox_head.reset_do_search(do_search)

    def reset_gumbel_buffer(self):
        return self.bbox_head.reset_gumbel_buffer()
