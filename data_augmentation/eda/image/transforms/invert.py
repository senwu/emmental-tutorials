from PIL import ImageOps

from eda.image.transforms.transform import EdaTransform


class Invert(EdaTransform):
    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        return ImageOps.invert(pil_img), label
