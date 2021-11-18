import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from emmental import EmmentalDataset


class CXR8Dataset(EmmentalDataset):
    """
    Dataset to load NIH Chest X-ray 14 dataset.

    Modified from reproduce-chexnet repo
    https://github.com/jrzech/reproduce-chexnet
    """

    def __init__(
        self,
        name,
        path_to_images,
        path_to_labels,
        split,
        transform=None,
        sample=0,
        finding="any",
    ):
        self.transform = transform
        self.path_to_images = path_to_images
        self.path_to_labels = path_to_labels
        self.split = split
        self.df = pd.read_csv(self.path_to_labels)
        self.df = self.df[self.df["fold"] == split]

        # can limit to sample, useful for testing
        # if split == "train" or split =="val": sample=500
        if sample > 0 and sample < len(self.df):
            self.df = self.df.sample(sample, random_state=self.seed)

        if (
            not finding == "any"
        ):  # can filter for positive findings of the kind described; for evaluation
            if finding in self.df.columns:
                if len(self.df[self.df[finding] == 1]) > 0:
                    self.df = self.df[self.df[finding] == 1]
                else:
                    print(
                        "No positive cases exist for "
                        + finding
                        + ", returning all unfiltered cases"
                    )
            else:
                print(
                    "cannot filter on finding "
                    + finding
                    + " as not in data - please check spelling"
                )

        self.df = self.df.set_index("Image Index")

        self.PRED_LABEL = [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ]

        X_dict = {"image_name": []}
        Y_dict = {}

        for idx in range(len(self.df)):
            X_dict["image_name"].append(self.df.index[idx])
            for label in self.PRED_LABEL:
                if label not in Y_dict:
                    Y_dict[label] = []
                Y_dict[label].append(self.df[label].iloc[idx].astype("int"))

        for label in self.PRED_LABEL:
            Y_dict[label] = torch.from_numpy(np.array(Y_dict[label]))

        super().__init__(name, X_dict=X_dict, Y_dict=Y_dict)

    def __len__(self):
        return len(self.X_dict["image_name"])

    def __getitem__(self, index):

        image = Image.open(
            os.path.join(self.path_to_images, self.X_dict["image_name"][index])
        )
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}

        x_dict["image"] = image

        return x_dict, y_dict
