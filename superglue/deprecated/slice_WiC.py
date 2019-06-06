import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# WARNING: Hardcoded here instead of Meta.config["learner_config"]["ignore_index"]
PADDING = 0

def slice_base(dataset):
    return torch.from_numpy(np.array([1] * len(dataset))), dataset.Y_dict["labels"]

def slice_verb(dataset):
    slice_name = "slice_verb"
    ind, pred = [], []
    cnt = 0
    for idx, pos in enumerate(dataset.X_dict["poses"]):
        if pos == "V":
            ind.append(1)
            pred.append(dataset.Y_dict["labels"][idx])
            cnt += 1
        else:
            ind.append(2)
            pred.append(PADDING)
    ind = torch.from_numpy(np.array(ind)).view(-1)
    pred = torch.from_numpy(np.array(pred)).view(-1)
    logger.info(f"Total {cnt} / {len(dataset)} in the slice {slice_name}")
    print(ind.size(), pred.size())
    return ind, pred

def slice_trigram(dataset):
    """The target word participates in the same trigram in both sentences"""
    def get_ngrams(tokens, window=1):
        num_ngrams = len(tokens) - window + 1
        for i in range(num_ngrams):
            yield tokens[i:i+window]    
    
    slice_name = "slice_trigram"
    ind = []
    pred = []
    cnt = 0
    labels = dataset.Y_dict["labels"]
    for idx, (target, sent1, sent2, sent1_idx, sent2_idx) in enumerate(zip(
        dataset.X_dict["words"], 
        dataset.X_dict["sent1"], 
        dataset.X_dict["sent2"],
        dataset.X_dict["sent1_ori_idxs"],
        dataset.X_dict["sent2_ori_idxs"],
        )):
        trigrams = []
        for sent, sent_idx in [(sent1, sent1_idx), (sent2, sent2_idx)]:
            tokens = sent.split()
            trigrams.append([' '.join(ngram).lower() 
                            for ngram in get_ngrams(tokens[sent_idx-2:sent_idx+2], window=3) 
                            if len(ngram) == 3])
        if (set(trigrams[0]).intersection(set(trigrams[1]))):
            cnt += 1
            ind.append(1)
            pred.append(labels[idx])
        else:
            ind.append(2)
            pred.append(PADDING)
            
    ind = torch.from_numpy(np.array(ind)).view(-1)
    pred = torch.from_numpy(np.array(pred)).view(-1)
    logger.info(f"Total {cnt} / {len(dataset)} in the slice {slice_name}")
    print(ind.size(), pred.size())
    return ind, pred

# def slice_mismatch(dataset):
#     """The target word has different form (e.g., tense, plurality) in both sentences"""
#     slice_name = "slice_mismatch"
#     ind = []
#     pred = []
#     cnt = 0
#     labels = dataset.Y_dict["labels"]
#     for idx, (target, sent1, sent2):
        

slice_func_dict = {
    "slice_base": slice_base,
    "slice_verb": slice_verb,
    "slice_trigram": slice_trigram,
}