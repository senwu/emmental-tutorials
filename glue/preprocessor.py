import codecs
import logging
import os

import numpy as np
import torch

from pytorch_pretrained_bert import BertTokenizer
from task_config import (
    INDEX_MAPPING,
    LABEL_MAPPING,
    SKIPPING_HEADER_MAPPING,
    SPLIT_MAPPING,
)

try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm


logger = logging.getLogger(__name__)

DELIMITER = "\t"


def preprocessor(
    data_dir,
    task_name,
    split,
    bert_model_name="bert-base-uncased",
    max_data_samples=None,
    max_sequence_length=128,
):

    sentences, labels = parse_tsv(data_dir, task_name, split, max_data_samples)

    labels = torch.from_numpy(np.array(labels))

    do_lower_case = "uncased" in bert_model_name

    tokenizer = BertTokenizer.from_pretrained(
        bert_model_name, do_lower_case=do_lower_case
    )

    bert_token_ids = []
    bert_token_masks = []
    bert_token_segments = []

    for sentence in sentences:
        if len(sentence) not in [1, 2]:
            logger.error("Sentence length doesn't match.")

        # Tokenize sentences
        tokenized_sentence = [tokenizer.tokenize(sent) for sent in sentence]
        sent1_tokens = tokenized_sentence[0]
        sent2_tokens = tokenized_sentence[1] if len(tokenized_sentence) == 2 else None

        # One sentence case
        if len(tokenized_sentence) == 1:
            # Remove tokens that exceeds the max_sequence_length
            if len(sent1_tokens) > max_sequence_length - 2:
                # Account for [CLS] and [SEP] with "- 2"
                sent1_tokens = sent1_tokens[: max_sequence_length - 2]
        # Two sentences case
        else:
            # Remove tokens that exceeds the max_sequence_length
            while True:
                total_length = len(sent1_tokens) + len(sent2_tokens)
                # Account for [CLS], [SEP], [SEP] with "- 3"
                if total_length <= max_sequence_length - 3:
                    break
                if len(sent1_tokens) > len(sent2_tokens):
                    sent1_tokens.pop()
                else:
                    sent2_tokens.pop()

        # Convert to BERT manner
        tokens = ["[CLS]"] + sent1_tokens + ["[SEP]"]
        token_segments = [0] * len(tokens)

        if sent2_tokens:
            tokens += sent2_tokens + ["[SEP]"]
            token_segments += [1] * (len(sent2_tokens) + 1)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Generate mask where 1 for real tokens and 0 for padding tokens
        token_masks = [1] * len(token_ids)

        # Append padding
        padding = [0] * (max_sequence_length - len(token_ids))

        token_ids += padding
        token_masks += padding
        token_segments += padding

        bert_token_ids.append(torch.LongTensor(token_ids))
        bert_token_masks.append(torch.LongTensor(token_masks))
        bert_token_segments.append(torch.LongTensor(token_segments))

    return bert_token_ids, bert_token_segments, bert_token_masks, labels


def parse_tsv(data_dir, task_name, split, max_data_samples=None):
    sentences = []
    labels = []

    tsv_path = os.path.join(data_dir, task_name, SPLIT_MAPPING[task_name][split])
    with codecs.open(tsv_path, "r", "utf-8") as f:
        # Skip header if needed
        if SKIPPING_HEADER_MAPPING[task_name]:
            f.readline()

        rows = list(enumerate(f))

        # Truncate to max_data_samples
        if max_data_samples:
            rows = rows[:max_data_samples]

        # Calculate the max number of column
        max_cloumns = len(rows[0][1].strip().split(DELIMITER))

        for idx, row in tqdm(rows):
            row = row.strip().split(DELIMITER)

            if len(row) > max_cloumns:
                logger.warning("Row has more columns than expected, skip...")
                continue

            sent1_idx, sent2_idx, label_idx = INDEX_MAPPING[task_name][split]

            if sent1_idx >= len(row) or sent2_idx >= len(row) or label_idx >= len(row):
                logger.warning("Data column doesn't match, skip...")
                continue

            sent1 = row[sent1_idx]
            sent2 = row[sent2_idx] if sent2_idx >= 0 else None

            if label_idx >= 0:
                if LABEL_MAPPING[task_name] is not None:
                    label = LABEL_MAPPING[task_name][row[label_idx]]
                else:
                    label = np.float32(row[label_idx])
            else:
                label = -1

            sentences.append([sent1] if sent2 is None else [sent1, sent2])
            labels.append(label)

    return sentences, labels


# Test purpose
if __name__ == "__main__":
    task_names = [
        "CoLA",
        "MNLI",
        # "MNLI_matched",
        # "MNLI_unmatched",
        "MRPC",
        "QNLI",
        "QQP",
        "RTE",
        "SNLI",
        "SST-2",
        "STS-B",
        "WNLI",
    ]

    splits = ["train", "dev", "test"]

    data_dir = "data"

    for task_name in task_names:
        for split in splits:
            print(task_name, split)
            print(
                preprocessor(
                    data_dir,
                    task_name,
                    split,
                    bert_model="bert-base-uncased",
                    max_data_samples=2,
                    max_sequence_length=20,
                )
            )
