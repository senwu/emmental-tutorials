import logging

from slicing.slicing_function import slicing_function

logger = logging.getLogger(__name__)

# WARNING: Hardcoded here instead of Meta.config["learner_config"]["ignore_index"]
PADDING = -100

@slicing_function()
def slice_base(example):
    return 1

@slicing_function(fields=["pos"])
def slice_verb(example):
    return example.pos == "V"

@slicing_function(fields=["word", "sentence1", "sentence2", "sentence1_idx", "sentence2_idx"])
def slice_trigram(example):
    def get_ngrams(tokens, window=1):
        num_ngrams = len(tokens) - window + 1
        for i in range(num_ngrams):
            yield tokens[i:i+window]    

    trigrams = []
    for sent, sent_idx in [(example.sentence1, example.sentence1_idx), 
                           (example.sentence2, example.sentence2_idx)]:
        tokens = sent.split()
        trigrams.append([' '.join(ngram).lower() 
                        for ngram in get_ngrams(tokens[sent_idx-2:sent_idx+2], window=3) 
                        if len(ngram) == 3])
    if (set(trigrams[0]).intersection(set(trigrams[1]))):
        return 1
        
slices = [
    slice_base,
    slice_verb,
    slice_trigram,
]

slice_func_dict = {slice.__name__: slice for slice in slices}