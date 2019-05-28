import json
import logging
import os

import numpy as np
import torch

from emmental.data import EmmentalDataLoader, EmmentalDataset
from task_config import SuperGLUE_LABEL_MAPPING, SuperGLUE_TASK_SPLIT_MAPPING
from tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


def get_WiC_dataloaders(
    data_dir,
    task_name="WiC",
    splits=["train", "val", "test"],
    max_data_samples=None,
    max_sequence_length=128,
    tokenizer_name="bert-base-uncased",
    batch_size=16,
    slice_func_dict = {},
):
    """Load WiC data and return dataloaders"""

    def parse_jsonl(jsonl_path):
#         print(jsonl_path)
        rows = [json.loads(row) for row in open(jsonl_path, encoding="utf-8")]
#         print(rows[0])
        # Truncate to max_data_samples
        if max_data_samples:
            rows = rows[:max_data_samples]

        uids = []
        sent1s = []
        sent2s = []
        sent1_idxs = []
        sent2_idxs = []
        sent1_ori_idxs = []
        sent2_ori_idxs = []
        labels = []
        words = []
        poses = []

        bert_token_ids = []
        bert_token_masks = []
        bert_token_segments = []

        max_len = -1
        
        for sample in rows:
            index = sample["idx"]
#             if index not in [3671, 2819, 3257, 2725]: continue
            sent1 = sample["sentence1"]
            sent2 = sample["sentence2"]
            word =  sample["word"]
            pos =  sample["pos"]
            sent1_idx = int(sample["sentence1_idx"])
            sent2_idx = int(sample["sentence2_idx"])
            sent1_ori_idx = int(sample["sentence1_idx"])
            sent2_ori_idx = int(sample["sentence1_idx"])
            label = sample["label"] if "label" in sample else True
            uids.append(index)
            sent1s.append(sent1)
            sent2s.append(sent2)
            sent1_ori_idxs.append(sent1_ori_idx)
            sent2_ori_idxs.append(sent2_ori_idx)
            words.append(word)
            poses.append(pos)
            labels.append(SuperGLUE_LABEL_MAPPING[task_name][label])

            # Tokenize sentences
            sent1_tokens = tokenizer.tokenize(sent1)
            sent2_tokens = tokenizer.tokenize(sent2)

            word_tokens_in_sent1 = tokenizer.tokenize(sent1.split()[sent1_idx])
            word_tokens_in_sent2 = tokenizer.tokenize(sent2.split()[sent2_idx])

            while True:
                total_length = len(sent1_tokens) + len(sent2_tokens)
                if total_length > max_len: max_len = total_length
                # Account for [CLS], [SEP], [SEP] with "- 3"
                if total_length <= max_sequence_length - 3:
                    break
                if len(sent1_tokens) > len(sent2_tokens):
                    sent1_tokens.pop()
                else:
                    sent2_tokens.pop()

            for idx in range(sent1_idx - 1, len(sent1_tokens)):
                if sent1_tokens[idx:idx+len(word_tokens_in_sent1)] == word_tokens_in_sent1:
                    sent1_idxs.append(idx + 1) # Add [CLS]
                    break
#             if (sent1_idx != sent1_idxs[-1]):
#                 print("word", sample["word"], sent1_idxs[-1], sent1, sent1_idx, sent1_tokens)
                
            for idx in range(sent2_idx - 1, len(sent2_tokens)):
                if sent2_tokens[idx:idx+len(word_tokens_in_sent2)] == word_tokens_in_sent2:
                    sent2_idxs.append(idx + len(sent1_tokens) + 2) # Add the length of the first sentence and [CLS] + [SEP]
                    break
#             if (sent2_idx != sent2_idxs[-1]):
#                 print("word", sample["word"], sent2_idxs[-1], sent2, sent2_idx, sent2_tokens)

    

            # Convert to BERT manner
            tokens = ["[CLS]"] + sent1_tokens + ["[SEP]"]
            token_segments = [0] * len(tokens)

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

        sent1_idxs = torch.from_numpy(np.array(sent1_idxs))
        sent2_idxs = torch.from_numpy(np.array(sent2_idxs))

        labels = torch.from_numpy(np.array(labels))
        print(f"max len {max_len}")
        return EmmentalDataset(
            name="SuperGLUE",
            X_dict={
                "uids": uids,
                "sent1": sent1s,
                "sent2": sent2s,
                "words": words,
                "poses": poses,
                "sent1_idxs": sent1_idxs,
                "sent2_idxs": sent2_idxs,
                "sent1_ori_idxs": sent1_ori_idxs,
                "sent2_ori_idxs": sent2_ori_idxs,
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
        
#         task_to_label_dict = {task_name: "labels"}
        task_to_label_dict = {}        
    
        for slice_name, slice_func in slice_func_dict.items():
            ind, pred = slice_func(dataset)
            dataset.Y_dict.update({f"{task_name}_slice_ind_{slice_name}": ind, f"{task_name}_slice_pred_{slice_name}": pred})
            task_to_label_dict.update({f"{task_name}_slice_ind_{slice_name}": f"{task_name}_slice_ind_{slice_name}", f"{task_name}_slice_pred_{slice_name}": f"{task_name}_slice_pred_{slice_name}"})
        
        task_to_label_dict.update({task_name: "labels"})

        dataloaders[split] = EmmentalDataLoader(
            task_to_label_dict=task_to_label_dict,
            dataset=dataset,
            split=split,
            batch_size=batch_size,
            shuffle=split == "True",
        )
        logger.info(f"Loaded {split} for {task_name}.")

    return dataloaders
