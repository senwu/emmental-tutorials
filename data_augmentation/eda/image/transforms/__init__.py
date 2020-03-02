from eda.image.transforms.auto_contrast import AutoContrast
from eda.image.transforms.blur import Blur
from eda.image.transforms.brightness import Brightness
from eda.image.transforms.center_crop import CenterCrop
from eda.image.transforms.color import Color
from eda.image.transforms.contrast import Contrast
from eda.image.transforms.cutout import Cutout
from eda.image.transforms.equalize import Equalize
from eda.image.transforms.horizontal_filp import HorizontalFlip
from eda.image.transforms.identity import Identity
from eda.image.transforms.invert import Invert
from eda.image.transforms.mixup import Mixup
from eda.image.transforms.posterize import Posterize
from eda.image.transforms.random_crop import RandomCrop
from eda.image.transforms.random_resize_crop import RandomResizedCrop
from eda.image.transforms.resize import Resize
from eda.image.transforms.rotate import Rotate
from eda.image.transforms.sharpness import Sharpness
from eda.image.transforms.shear_x import ShearX
from eda.image.transforms.shear_y import ShearY
from eda.image.transforms.smooth import Smooth
from eda.image.transforms.solarize import Solarize
from eda.image.transforms.translate_x import TranslateX
from eda.image.transforms.translate_y import TranslateY
from eda.image.transforms.vertical_flip import VerticalFlip

ALL_TRANSFORMS = {
    "AutoContrast": AutoContrast,
    "Blur": Blur,
    "Brightness": Brightness,
    "CenterCrop": CenterCrop,
    "Color": Color,
    "Contrast": Contrast,
    "Cutout": Cutout,
    "Equalize": Equalize,
    "HorizontalFlip": HorizontalFlip,
    "Identity": Identity,
    "Invert": Invert,
    "Mixup": Mixup,
    "Posterize": Posterize,
    "RandomCrop": RandomCrop,
    "RandomResizedCrop": RandomResizedCrop,
    "Resize": Resize,
    "Rotate": Rotate,
    "Sharpness": Sharpness,
    "ShearX": ShearX,
    "ShearY": ShearY,
    "Smooth": Smooth,
    "Solarize": Solarize,
    "TranslateX": TranslateX,
    "TranslateY": TranslateY,
    "VerticalFlip": VerticalFlip,
}
