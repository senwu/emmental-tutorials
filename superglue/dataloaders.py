import logging
import os

import augmentation
import parsers
from task_config import SuperGLUE_TASK_SPLIT_MAPPING
from tokenizer import get_tokenizer

from emmental.data import EmmentalDataLoader

logger = logging.getLogger(__name__)


def get_dataloaders(
    data_dir,
    task_name="MultiRC",
    splits=["train", "val", "test"],
    max_data_samples=None,
    max_sequence_length=256,
    tokenizer_name="bert-base-uncased",
    batch_size=16,
    augment=False,
    uid="uids",
):
    """Load data and return dataloaders."""

    dataloaders = []

    tokenizer = get_tokenizer(tokenizer_name)

    for split in splits:
        jsonl_path = os.path.join(
            data_dir, task_name, SuperGLUE_TASK_SPLIT_MAPPING[task_name][split]
        )
        dataset = parsers.parser[task_name](
            jsonl_path, tokenizer, uid, max_data_samples, max_sequence_length
        )
        dataloader = EmmentalDataLoader(
            task_to_label_dict={task_name: "labels"},
            dataset=dataset,
            split=split,
            batch_size=batch_size,
            shuffle=(split == "train"),
        )
        dataloaders.append(dataloader)

        if (
            augment
            and split == "train"
            and task_name in augmentation.augmentation_funcs
        ):
            augmentation_funcs = augmentation.augmentation_funcs[task_name]
            for af in augmentation_funcs:
                dataset = af(dataset)
                dataloader = EmmentalDataLoader(
                    task_to_label_dict={task_name: "labels"},
                    dataset=dataset,
                    split=split,
                    batch_size=batch_size,
                    shuffle=(split == "train"),
                )
                dataloaders.append(dataloader)

        logger.info(f"Loaded {split} for {task_name} with {len(dataset)} samples.")

    return dataloaders
