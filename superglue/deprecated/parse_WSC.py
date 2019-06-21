import json
import logging
import os

import numpy as np
import torch

from emmental.data import EmmentalDataLoader, EmmentalDataset
from task_config import SuperGLUE_LABEL_MAPPING, SuperGLUE_TASK_SPLIT_MAPPING
from tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


def get_WSC_dataloaders(
    data_dir,
    task_name="WSC",
    splits=["train", "val", "test"],
    max_data_samples=None,
    max_sequence_length=256,
    tokenizer_name="bert-base-uncased",
    batch_size=16,
):
    """Load WSC DATA AND return dataloaders"""

    def parse_jsonl(jsonl_path):
        print(jsonl_path)
        rows = [json.loads(row) for row in open(jsonl_path, encoding="utf-8")]
        print(rows[0])
        # Truncate to max_data_samples
        if max_data_samples:
            rows = rows[:max_data_samples]

        uids = []
        sents = []
        span1s = []
        span2s = []
        span1_idxs = []
        span2_idxs = []
        labels = []

        bert_token_ids = []
        bert_token_masks = []
        bert_token_segments = []

        max_len = -1
        
        for sample in rows:
            index = sample["idx"]
            sent = sample["text"]
            span1 = sample["target"]["span1_text"].strip()
            span2 = sample["target"]["span2_text"].strip()
            span1_idx = sample["target"]["span1"]
            span2_idx = sample["target"]["span2"]
#             if index != 78: continue
            assert span1_idx[0]<= span1_idx[1]
            assert span2_idx[0]<= span2_idx[1]
            
            label = sample["label"] if "label" in sample else True
            
            uids.append(index)
            sents.append(sent)
            span1s.append(span1)
            span2s.append(span2)
            labels.append(SuperGLUE_LABEL_MAPPING[task_name][label])

            text = sent.split()
            
            # Tokenize sentences
            sent_tokens = sent.split()
            span1_idxs.append([span1_idx[0]+1, span1_idx[1]])
            span2_idxs.append([span2_idx[0]+1, span2_idx[1]])

            # Convert to BERT manner
            tokens = ["[CLS]"] + sent_tokens + ["[SEP]"]

            if len(tokens) > max_len:
                max_len = len(tokens)
            
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            token_segments = [0] * len(token_ids)
            # Generate mask where 1 for real tokens and 0 for padding tokens
            token_masks = [1] * len(token_ids)

            bert_token_ids.append(torch.LongTensor(token_ids))
            bert_token_masks.append(torch.LongTensor(token_masks))
            bert_token_segments.append(torch.LongTensor(token_segments))

        span1_idxs = torch.from_numpy(np.array(span1_idxs))
        span2_idxs = torch.from_numpy(np.array(span2_idxs))

        labels = torch.from_numpy(np.array(labels))
        print(f"max len {max_len}")
        return EmmentalDataset(
            name="SuperGLUE",
            X_dict={
                "uids": uids,
                "sent": sents,
                "span1": span1s,
                "span2": span2s,
                "span1_idxs": span1_idxs,
                "span2_idxs": span2_idxs,
                "token_ids": bert_token_ids,
                "token_masks": bert_token_masks,
                "token_segments": bert_token_segments,
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
            shuffle= split == "train",
        )
        logger.info(f"Loaded {split} for {task_name}. {split == 'train'}")

    return dataloaders
