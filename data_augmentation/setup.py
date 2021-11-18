from setuptools import find_packages, setup

setup(
    name="EDA",
    version="0.0.1",
    description="Understanding the data augmentation using Emmnetal.",
    install_requires=[
        "emmental>=0.1.0,<0.2.0",
        "pillow>=8.3.2,<9.0.0",
        "torchvision>=0.4.2,<1.0.0",
    ],
    scripts=["bin/image"],
    packages=find_packages(),
)
