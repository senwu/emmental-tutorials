# Copyright (c) 2021 Sen Wu. All Rights Reserved.


import random

import numpy as np
import pandas as pd
import torch
from emmental.data import EmmentalDataset
from tqdm import tqdm

from utils import read_csv


class ToxicCommentDataset(EmmentalDataset):
    """Dataset to load Toxic comment dataset."""

    def __init__(
        self,
        name,
        file_path,
        id_file_path,
        split,
        tokenizer,
        max_data_samples=None,
        max_length=128,
    ):
        df = pd.read_csv(file_path)
        ids = read_csv(id_file_path)
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_data_samples = max_data_samples
        self.label_keys = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]

        X_dict = {"id": [], "comment_text": []}
        Y_dict = {"labels": []}

        if self.max_data_samples:
            random.shuffle(ids)

        for i in tqdm(range(len(df))):
            if df.iloc[i]["id"] not in ids:
                continue
            for key in X_dict.keys():
                X_dict[key].append(df.iloc[i][key])
            Y_dict["labels"].append([float(df.iloc[i][key]) for key in self.label_keys])
            if self.max_data_samples and len(X_dict["id"]) >= self.max_data_samples:
                break
        for key in Y_dict.keys():
            Y_dict[key] = torch.from_numpy(np.array(Y_dict[key]))

        super().__init__(name, X_dict=X_dict, Y_dict=Y_dict, uid="id")

    def __len__(self):
        return len(self.X_dict["id"])

    def __getitem__(self, index):
        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}
        # print(x_dict, y_dict)
        inputs = self.tokenizer.encode_plus(
            x_dict["comment_text"],
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        x_dict.update({f"feat_{key}": value[0] for key, value in inputs.items()})
        return x_dict, y_dict
