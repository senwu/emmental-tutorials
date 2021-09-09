import logging

import torch
from dataloader import (
    cv_split,
    cv_split2,
    load_CR,
    load_MPQA,
    load_MR,
    load_SST,
    load_SUBJ,
    load_TREC,
)

from emmental.data import EmmentalDataLoader, EmmentalDataset

logger = logging.getLogger(__name__)


load_dataset = {
    "mr": load_MR,
    "subj": load_SUBJ,
    "cr": load_CR,
    "mpqa": load_MPQA,
    "trec": load_TREC,
    "sst": load_SST,
}


def load_data(data_dir, task, cv, training_data_size=None):
    # Load data from file
    if task == "trec":
        train_x, train_y, test_x, test_y = load_dataset[task](data_dir)
        data = train_x + test_x
        label = None
    elif task == "sst":
        train_x, train_y, valid_x, valid_y, test_x, test_y = load_dataset[task](
            data_dir
        )
        data = train_x + valid_x + test_x
        label = None
    else:
        data, label = load_dataset[task](data_dir)

    if task == "trec":
        train_x, train_y, valid_x, valid_y = cv_split2(
            train_x, train_y, nfold=10, valid_id=cv
        )
    elif task != "sst":
        train_x, train_y, valid_x, valid_y, test_x, test_y = cv_split(
            data, label, nfold=10, test_id=cv
        )

    nclasses = max(train_y) + 1

    # Set the training dataset size
    if training_data_size:
        train_x = train_x[:training_data_size]
        train_y = train_y[:training_data_size]

    train_y = torch.LongTensor(train_y)
    valid_y = torch.LongTensor(valid_y)
    test_y = torch.LongTensor(test_y)

    dataset = {
        "train": (train_x, train_y),
        "valid": (valid_x, valid_y),
        "test": (test_x, test_y),
        "nclasses": nclasses,
    }

    return dataset, data


def create_dataloaders(task_name, dataset, batch_size, word2id, oov="~#OoV#~"):
    # Create dataloaders
    oov_id = word2id[oov]
    dataloaders = []

    for split in ["train", "valid", "test"]:
        split_x, split_y = dataset[split]
        split_x = [
            torch.LongTensor([word2id.get(w, oov_id) for w in seq]) for seq in split_x
        ]

        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict={task_name: "label"},
                dataset=EmmentalDataset(
                    name=task_name,
                    X_dict={"feature": split_x},
                    Y_dict={"label": split_y},
                ),
                split=split,
                batch_size=batch_size,
                shuffle=True if split == "train" else False,
            )
        )
        logger.info(
            f"Loaded {split} for {task_name} containing {len(split_x)} samples."
        )

    return dataloaders
