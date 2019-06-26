import json
import logging
import sys

import numpy as np
import torch
from emmental.data import EmmentalDataset
from task_config import SuperGLUE_LABEL_MAPPING

sys.path.append("..")  # Adds higher directory to python modules path.


logger = logging.getLogger(__name__)

TASK_NAME = "COPA"


def parse(jsonl_path, tokenizer, uid, max_data_samples, max_sequence_length):
    logger.info(f"Loading data from {jsonl_path}.")
    rows = [json.loads(row) for row in open(jsonl_path, encoding="utf-8")]
    for i in range(2):
        logger.info(f"Sample {i}: {rows[i]}")

    # Truncate to max_data_samples
    if max_data_samples:
        rows = rows[:max_data_samples]
        logger.info(f"Truncating to {max_data_samples} samples.")

    # unique ids
    uids = []
    # sentence1
    sent1s = []
    # sentence2
    sent2s = []
    # choice1
    choice1s = []
    # choice2
    choice2s = []

    labels = []

    bert_token1_ids = []
    bert_token2_ids = []

    bert_token1_masks = []
    bert_token2_masks = []

    bert_token1_segments = []
    bert_token2_segments = []

    # Check the maximum token length
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
        labels.append(SuperGLUE_LABEL_MAPPING[TASK_NAME][label])

        # Tokenize sentences
        sent1_tokens = tokenizer.tokenize(sent1)
        sent2_tokens = tokenizer.tokenize(sent2)

        # Tokenize choices
        choice1_tokens = tokenizer.tokenize(choice1)
        choice2_tokens = tokenizer.tokenize(choice2)

        # Convert to BERT manner
        tokens1 = (
            ["[CLS]"]
            + sent1_tokens
            + ["[SEP]"]
            + sent2_tokens
            + ["[SEP]"]
            + choice1_tokens
            + ["[SEP]"]
        )
        tokens2 = (
            ["[CLS]"]
            + sent1_tokens
            + ["[SEP]"]
            + sent2_tokens
            + ["[SEP]"]
            + choice2_tokens
            + ["[SEP]"]
        )

        token1_ids = tokenizer.convert_tokens_to_ids(tokens1)
        token2_ids = tokenizer.convert_tokens_to_ids(tokens2)

        padding1 = [0] * (max_sequence_length - len(token1_ids))
        padding2 = [0] * (max_sequence_length - len(token2_ids))

        token1_masks = [1] * len(token1_ids)
        token2_masks = [1] * len(token2_ids)

        token1_segments = [0] * len(token1_ids)
        token2_segments = [0] * len(token2_ids)

        token1_ids += padding1
        token2_ids += padding2

        token1_masks += padding1
        token2_masks += padding2

        token1_segments += padding1
        token2_segments += padding2

        if len(token1_ids) > max_len:
            max_len = len(token1_ids)
        if len(token2_ids) > max_len:
            max_len = len(token2_ids)

        bert_token1_ids.append(torch.LongTensor(token1_ids))
        bert_token2_ids.append(torch.LongTensor(token2_ids))

        bert_token1_masks.append(torch.LongTensor(token1_masks))
        bert_token2_masks.append(torch.LongTensor(token2_masks))

        bert_token1_segments.append(torch.LongTensor(token1_segments))
        bert_token2_segments.append(torch.LongTensor(token2_segments))

    labels = torch.from_numpy(np.array(labels))

    logger.info(f"Max token len {max_len}")

    return EmmentalDataset(
        name="SuperGLUE",
        uid="uids",
        X_dict={
            "uids": uids,
            "sentence1": sent1s,
            "sentence2": sent2s,
            "choice1": choice1s,
            "choice2": choice2s,
            "token1_ids": bert_token1_ids,
            "token2_ids": bert_token2_ids,
            "token1_masks": bert_token1_masks,
            "token2_masks": bert_token2_masks,
            "token1_segments": bert_token1_segments,
            "token2_segments": bert_token2_segments,
        },
        Y_dict={"labels": labels},
    )
