from eda.image.transforms.transform import EdaTransform
from PIL import ImageOps


class Equalize(EdaTransform):
    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        return ImageOps.equalize(pil_img), label
