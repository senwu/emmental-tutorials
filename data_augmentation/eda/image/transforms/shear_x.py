import random

from eda.image.transforms.transform import EdaTransform
from eda.image.transforms.utils import categorize_value
from PIL import Image


class ShearX(EdaTransform):

    value_range = (0.0, 0.3)

    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        degree = categorize_value(self.level, self.value_range, "float")
        if random.random() > 0.5:
            degree = -degree
        return (
            pil_img.transform(pil_img.size, Image.AFFINE, (1, degree, 0, 0, 1, 0)),
            label,
        )
