import torchvision.transforms as transforms

from eda.image.transforms.transform import EdaTransform


class ToTensor(EdaTransform):
    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        return transforms.ToTensor()(pil_img), label
