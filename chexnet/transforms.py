from torchvision import transforms


def get_data_transforms(dataset_name):
    """Get data transofrms based on dataset name."""
    data_transforms = None

    if "CXR8" in dataset_name:
        # use imagenet mean,std for normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # define torchvision transforms
        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(224),
                    # because scale doesn't always give 224 x 224, this ensures 224 x
                    # 224
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }

    return data_transforms
