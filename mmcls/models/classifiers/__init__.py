# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import ImageClassifier
from ..my_reprensentor.representor import ImageRepresentor

__all__ = ['BaseClassifier', 'ImageClassifier', 'ImageRepresentor']
