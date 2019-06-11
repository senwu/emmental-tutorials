import logging

import torch

from augmentation.augmentation_function import augmentation_function

logger = logging.getLogger(__name__)

@augmentation_function()
def identity_map(x_dict, y_dict):
    return x_dict, y_dict

@augmentation_function()
def flip_sentence_order(x_dict, y_dict):
    x = x_dict
    y = y_dict
    x["sentence1"], x["sentence2"] = x["sentence2"], x["sentence1"]
    x["sentence1_idx"], x["sentence2_idx"] = x["sentence2_idx"], x["sentence1_idx"]
    x["token1_idx"], x["token2_idx"] = x["token2_idx"], x["token1_idx"]
    CLS = x["token_ids"][0].view(-1)
    SEP = x["token_ids"][-1].view(-1)
    sep_index = (x["token_ids"] == SEP.item()).nonzero()[0].item()
    sent1_tokens = x["token_ids"][1:sep_index]
    sent2_tokens = x["token_ids"][sep_index+1:-1]
    x["token_ids"] = torch.cat([CLS, sent2_tokens, SEP, sent1_tokens, SEP], dim=0)
    zero_segments = torch.zeros(1 + len(sent2_tokens) + 1, dtype=torch.long)
    one_segments = torch.ones(len(sent1_tokens) + 1, dtype=torch.long)
    x["token_segments"] = torch.cat([zero_segments, one_segments])
    return x, y

augmentation_funcs = [
    flip_sentence_order,
    identity_map,
]