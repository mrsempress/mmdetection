import numpy as np

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class QrCodeDataset(CustomDataset):
    CLASSES = ('qr')