import logging
import os

from task_config import SuperGLUE_TASK_SPLIT_MAPPING
from tokenizer import get_tokenizer

import parsers
from emmental.data import EmmentalDataLoader

logger = logging.getLogger(__name__)


def get_dataloaders(
    data_dir,
    task_name="MultiRC",
    splits=["train", "val", "test"],
    max_data_samples=None,
    max_sequence_length=128,
    tokenizer_name="bert-base-uncased",
    batch_size=16,
    uid="uids",
):
    """Load data and return dataloaders"""

    dataloaders = []

    tokenizer = get_tokenizer(tokenizer_name)

    for split in splits:
        jsonl_path = os.path.join(
            data_dir, task_name, SuperGLUE_TASK_SPLIT_MAPPING[task_name][split]
        )
        dataset = parsers.parser[task_name](
            jsonl_path, tokenizer, uid, max_data_samples, max_sequence_length
        )

        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict={task_name: "labels"},
                dataset=dataset,
                split=split,
                batch_size=batch_size,
                shuffle=split == "train",
            )
        )
        logger.info(f"Loaded {split} for {task_name} with {len(dataset)} samples.")

    return dataloaders
