# Copyright (c) 2021 Sen Wu. All Rights Reserved.


import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from emmental.data import EmmentalDataset


class MultiLabelDataset(EmmentalDataset):
    """Dataset to load multi-label dataset."""

    def __init__(
        self,
        name,
        data_path,
        input_field,
        label_fields,
        split,
        tokenizer,
        max_data_samples=None,
        max_seq_length=128,
    ):
        self.name = name
        self.uid = "_uids_"
        self.df = pd.read_csv(data_path)
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_data_samples = max_data_samples
        self.input_field = input_field
        self.label_fields = label_fields

        if self.max_data_samples:
            self.df = self.df.sample(frac=1)

    def __len__(self):
        return len(self.df) if self.max_data_samples is None else self.max_data_samples

    def __getitem__(self, index):
        x_dict = {
            self.input_field: self.df.iloc[index][self.input_field],
            self.uid: f"{self.name}_{index}",
        }
        y_dict = {
            "labels": torch.from_numpy(
                np.array([float(self.df.iloc[index][key]) for key in self.label_fields])
            )
        }

        inputs = self.tokenizer.encode_plus(
            x_dict[self.input_field],
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        x_dict.update({f"feat_{key}": value[0] for key, value in inputs.items()})

        return x_dict, y_dict
