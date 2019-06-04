import json
import logging
import os

import numpy as np
import torch

from emmental.data import EmmentalDataLoader, EmmentalDataset
from task_config import SuperGLUE_LABEL_MAPPING, SuperGLUE_TASK_SPLIT_MAPPING
from tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


def get_COPA_dataloaders(
    data_dir,
    task_name="COPA",
    splits=["train", "val", "test"],
    max_data_samples=None,
    max_sequence_length=128,
    tokenizer_name="bert-base-uncased",
    batch_size=16,
):
    """Load COPA DATA AND return dataloaders"""

    def parse_jsonl(jsonl_path):
        print(jsonl_path)
        rows = [json.loads(row) for row in open(jsonl_path, encoding="utf-8")]
        print(rows[0])
        # Truncate to max_data_samples
        if max_data_samples:
            rows = rows[:max_data_samples]

        uids = []
        sent1s = []
        sent2s = []
        choice1s = []
        choice2s = []
        labels = []

        bert_token1_ids = []
        bert_token2_ids = []

        max_len = -1
        
        for sample in rows:
            index = sample["idx"]
            sent1 = sample["premise"]
            sent2 = sample["question"]
            
            sent2 = (
                    "What was the cause of this?"
                    if sent2 == "cause"
                    else "What happened as a result?"
                )
            
            choice1 = sample["choice1"]
            choice2 = sample["choice2"]
            label = sample["label"] if "label" in sample else True
            uids.append(index)
            sent1s.append(sent1)
            sent2s.append(sent2)
            choice1s.append(choice1)
            choice2s.append(choice2)
            labels.append(SuperGLUE_LABEL_MAPPING[task_name][label])

            # Tokenize sentences
            sent1_tokens = tokenizer.tokenize(sent1)
            sent2_tokens = tokenizer.tokenize(sent2)

            # Tokenize choices
            choice1_tokens = tokenizer.tokenize(choice1)
            choice2_tokens = tokenizer.tokenize(choice2)

            # Convert to BERT manner
            tokens1 = ["[CLS]"] + sent1_tokens + ["[SEP]"] + sent2_tokens + ["[SEP]"] + choice1_tokens + ["[SEP]"]
            tokens2 = ["[CLS]"] + sent1_tokens + ["[SEP]"] + sent2_tokens + ["[SEP]"] + choice2_tokens + ["[SEP]"]

            token1_ids = tokenizer.convert_tokens_to_ids(tokens1)
            token2_ids = tokenizer.convert_tokens_to_ids(tokens2)

            if len(token1_ids) > max_len: max_len = len(token1_ids)
            if len(token2_ids) > max_len: max_len = len(token2_ids)

            
            token1_ids += [0] * (max_sequence_length - len(token1_ids))
            token2_ids += [0] * (max_sequence_length - len(token2_ids))
            
            bert_token1_ids.append(torch.LongTensor(token1_ids))
            bert_token2_ids.append(torch.LongTensor(token2_ids))

        labels = torch.from_numpy(np.array(labels))
        print(f"max len {max_len}")
        return EmmentalDataset(
            name="SuperGLUE",
            X_dict={
                "uids": uids,
                "sent1": sent1s,
                "sent2": sent2s,
                "token1_ids": bert_token1_ids,
                "token2_ids": bert_token2_ids,
            },
            Y_dict={"labels": labels},
        )

    dataloaders = {}

    tokenizer = get_tokenizer(tokenizer_name)

    for split in splits:
        jsonl_path = os.path.join(
            data_dir, task_name, SuperGLUE_TASK_SPLIT_MAPPING[task_name][split]
        )
        dataset = parse_jsonl(jsonl_path)

        dataloaders[split] = EmmentalDataLoader(
            task_to_label_dict={task_name: "labels"},
            dataset=dataset,
            split=split,
            batch_size=batch_size,
            shuffle=split == "True",
        )
        logger.info(f"Loaded {split} for {task_name}.")

    return dataloaders
