from numpy import random
from PIL import ImageFilter
import torchvision.transforms.transforms as transforms

from ..registry import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None


@PIPELINES.register_module
class ImgRandomResizedCrop(transforms.RandomResizedCrop):

    def __init__(self, *args, **kwargs):
        super(ImgRandomResizedCrop, self).__init__(*args, **kwargs)


@PIPELINES.register_module
class ImgColorJitter(transforms.ColorJitter):

    def __init__(self, *args, **kwargs):
        super(ImgColorJitter, self).__init__(*args, **kwargs)


@PIPELINES.register_module
class ImgRandomGrayscale(transforms.RandomGrayscale):

    def __init__(self, *args, **kwargs):
        super(ImgRandomGrayscale, self).__init__(*args, **kwargs)


@PIPELINES.register_module
class ImgGaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(.1, 2.)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


@PIPELINES.register_module
class ImgRandomApplyColorJitter(transforms.RandomApply):

    def __init__(self, *args, p=0.5, **kwargs):
        super(ImgRandomApplyColorJitter, self).__init__([
            transforms.ColorJitter(*args, **kwargs)
        ], p=p)


@PIPELINES.register_module
class ImgRandomApplyGaussianBlur(transforms.RandomApply):

    def __init__(self, *args, p=0.5, **kwargs):
        super(ImgRandomApplyGaussianBlur, self).__init__([
            ImgGaussianBlur(*args, **kwargs)
        ], p=p)


@PIPELINES.register_module
class ImgRandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __init__(self, **kwargs):
        super(ImgRandomHorizontalFlip, self).__init__(**kwargs)


@PIPELINES.register_module
class ImgToTensor(transforms.ToTensor):

    def __init__(self):
        super(ImgToTensor, self).__init__()


@PIPELINES.register_module
class ImgNormalize(transforms.Normalize):

    def __init__(self, **kwargs):
        super(ImgNormalize, self).__init__(**kwargs)

