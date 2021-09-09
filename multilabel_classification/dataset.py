# Copyright (c) 2021 Sen Wu. All Rights Reserved.


import numpy as np
import pandas as pd
import torch
from emmental.data import EmmentalDataset
from tqdm import tqdm


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
        df = pd.read_csv(data_path)
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_data_samples = max_data_samples
        self.input_field = input_field
        self.label_fields = label_fields

        X_dict = {self.input_field: []}
        Y_dict = {"labels": []}

        if self.max_data_samples:
            df = df.sample(frac=1)

        for i in tqdm(range(len(df))):
            X_dict[self.input_field].append(df.iloc[i][self.input_field])
            Y_dict["labels"].append(
                [float(df.iloc[i][key]) for key in self.label_fields]
            )
            if (
                self.max_data_samples
                and len(X_dict[self.input_field]) >= self.max_data_samples
            ):
                break
        for key in Y_dict.keys():
            Y_dict[key] = torch.from_numpy(np.array(Y_dict[key]))

        super().__init__(name, X_dict=X_dict, Y_dict=Y_dict)

    def __len__(self):
        return len(self.X_dict[self.input_field])

    def __getitem__(self, index):
        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}

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
