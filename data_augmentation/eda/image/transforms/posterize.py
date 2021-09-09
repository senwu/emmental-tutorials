from eda.image.transforms.transform import EdaTransform
from eda.image.transforms.utils import categorize_value
from PIL import ImageOps


class Posterize(EdaTransform):

    value_range = (0, 4)

    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        degree = categorize_value(self.level, self.value_range, "int")
        return ImageOps.posterize(pil_img, degree), label
