# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from torchvision.datasets.folder import ImageFolder
import torchvision.transforms.transforms as transforms

from ..datasets.pipelines.compose import Compose
from .registry import DATASETS


@DATASETS.register_module
class TwoCropsDataset(ImageFolder):

    def __init__(self, img_prefix, pipeline):
        pipeline = Compose(pipeline)
        transform=TwoCropsTransform(pipeline)
        super(TwoCropsDataset, self).__init__(img_prefix, transform=transform)
        self.flag = np.zeros(len(self), dtype=np.uint8)  # to fit mmdetection settings

    def __getitem__(self, index):
        sample, _ = super().__getitem__(index)
        return sample


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform
        self.to_tensor = transforms.ToTensor()

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        # return dict(img=self.to_tensor(x), img_q=q, img_k=k)
        return dict(img=q, img_k=k)
