import logging
import sys

import numpy as np
import pandas as pd
import torch
from emmental.data import EmmentalDataset
from task_config import SuperGLUE_LABEL_MAPPING

sys.path.append("..")  # Adds higher directory to python modules path.


logger = logging.getLogger(__name__)

TASK_NAME = "SWAG"


def parse(csv_path, tokenizer, uid, max_data_samples, max_sequence_length):
    logger.info(f"Loading data from {csv_path}.")
    rows = pd.read_csv(csv_path)

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
    # choice3
    choice3s = []
    # choice4
    choice4s = []

    labels = []

    bert_token1_ids = []
    bert_token2_ids = []
    bert_token3_ids = []
    bert_token4_ids = []

    bert_token1_masks = []
    bert_token2_masks = []
    bert_token3_masks = []
    bert_token4_masks = []

    bert_token1_segments = []
    bert_token2_segments = []
    bert_token3_segments = []
    bert_token4_segments = []

    # Check the maximum token length
    max_len = -1

    for ex_idx, ex in rows.iterrows():
        sent1 = ex["sent1"]
        sent2 = ex["sent2"]

        choice1 = ex["ending0"]
        choice2 = ex["ending1"]
        choice3 = ex["ending2"]
        choice4 = ex["ending3"]

        label = ex["label"] if "label" in ex else 0

        uids.append(ex_idx)
        sent1s.append(sent1)
        sent2s.append(sent2)
        choice1s.append(choice1)
        choice2s.append(choice2)
        choice3s.append(choice3)
        choice4s.append(choice4)

        labels.append(SuperGLUE_LABEL_MAPPING[TASK_NAME][label])

        # Tokenize sentences
        sent1_tokens = tokenizer.tokenize(sent1)
        sent2_tokens = tokenizer.tokenize(sent2)
        choice1_tokens = tokenizer.tokenize(choice1)
        choice2_tokens = tokenizer.tokenize(choice2)
        choice3_tokens = tokenizer.tokenize(choice3)
        choice4_tokens = tokenizer.tokenize(choice4)

        # Convert to BERT manner
        token1 = (
            ["[CLS]"]
            + sent1_tokens
            + ["[SEP]"]
            + sent2_tokens
            + choice1_tokens
            + ["[SEP]"]
        )
        token2 = (
            ["[CLS]"]
            + sent1_tokens
            + ["[SEP]"]
            + sent2_tokens
            + choice2_tokens
            + ["[SEP]"]
        )
        token3 = (
            ["[CLS]"]
            + sent1_tokens
            + ["[SEP]"]
            + sent2_tokens
            + choice3_tokens
            + ["[SEP]"]
        )
        token4 = (
            ["[CLS]"]
            + sent1_tokens
            + ["[SEP]"]
            + sent2_tokens
            + choice4_tokens
            + ["[SEP]"]
        )

        max_choice_len = 0

        token1_ids = tokenizer.convert_tokens_to_ids(token1)[:max_sequence_length]
        token2_ids = tokenizer.convert_tokens_to_ids(token2)[:max_sequence_length]
        token3_ids = tokenizer.convert_tokens_to_ids(token3)[:max_sequence_length]
        token4_ids = tokenizer.convert_tokens_to_ids(token4)[:max_sequence_length]

        token1_masks = [1] * len(token1_ids)
        token2_masks = [1] * len(token2_ids)
        token3_masks = [1] * len(token3_ids)
        token4_masks = [1] * len(token4_ids)

        token1_segments = [0] * len(token1_ids)
        token2_segments = [0] * len(token2_ids)
        token3_segments = [0] * len(token3_ids)
        token4_segments = [0] * len(token4_ids)

        if len(token1_ids) > max_len:
            max_len = len(token1_ids)
        if len(token2_ids) > max_len:
            max_len = len(token2_ids)
        if len(token3_ids) > max_len:
            max_len = len(token3_ids)
        if len(token4_ids) > max_len:
            max_len = len(token4_ids)

        max_choice_len = max(max_choice_len, len(token1_ids))
        max_choice_len = max(max_choice_len, len(token2_ids))
        max_choice_len = max(max_choice_len, len(token3_ids))
        max_choice_len = max(max_choice_len, len(token4_ids))

        token1_ids += [0] * (max_choice_len - len(token1_ids))
        token2_ids += [0] * (max_choice_len - len(token2_ids))
        token3_ids += [0] * (max_choice_len - len(token3_ids))
        token4_ids += [0] * (max_choice_len - len(token4_ids))

        token1_masks += [0] * (max_choice_len - len(token1_masks))
        token2_masks += [0] * (max_choice_len - len(token2_masks))
        token3_masks += [0] * (max_choice_len - len(token3_masks))
        token4_masks += [0] * (max_choice_len - len(token4_masks))

        token1_segments += [0] * (max_choice_len - len(token1_segments))
        token2_segments += [0] * (max_choice_len - len(token2_segments))
        token3_segments += [0] * (max_choice_len - len(token3_segments))
        token4_segments += [0] * (max_choice_len - len(token4_segments))

        bert_token1_ids.append(torch.LongTensor(token1_ids))
        bert_token2_ids.append(torch.LongTensor(token2_ids))
        bert_token3_ids.append(torch.LongTensor(token3_ids))
        bert_token4_ids.append(torch.LongTensor(token4_ids))

        bert_token1_masks.append(torch.LongTensor(token1_masks))
        bert_token2_masks.append(torch.LongTensor(token2_masks))
        bert_token3_masks.append(torch.LongTensor(token3_masks))
        bert_token4_masks.append(torch.LongTensor(token4_masks))

        bert_token1_segments.append(torch.LongTensor(token1_segments))
        bert_token2_segments.append(torch.LongTensor(token2_segments))
        bert_token3_segments.append(torch.LongTensor(token3_segments))
        bert_token4_segments.append(torch.LongTensor(token4_segments))

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
            "choice3": choice3s,
            "choice4": choice4s,
            "token1_ids": bert_token1_ids,
            "token2_ids": bert_token2_ids,
            "token3_ids": bert_token3_ids,
            "token4_ids": bert_token4_ids,
            "token1_masks": bert_token1_masks,
            "token2_masks": bert_token2_masks,
            "token3_masks": bert_token3_masks,
            "token4_masks": bert_token4_masks,
            "token1_segments": bert_token1_segments,
            "token2_segments": bert_token2_segments,
            "token3_segments": bert_token3_segments,
            "token4_segments": bert_token4_segments,
        },
        Y_dict={"labels": labels},
    )
