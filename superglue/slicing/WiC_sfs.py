import logging

from slicing.slicing_function import slicing_function

logger = logging.getLogger(__name__)


@slicing_function()
def slice_base(example):
    return 1


@slicing_function(fields=["pos"])
def slice_verb(example):
    """Is the target word a verb."""
    return example.pos == "V"


@slicing_function(fields=["pos"])
def slice_noun(example):
    """Is the target word a noun."""
    return example.pos == "N"


@slicing_function(
    fields=["word", "sentence1", "sentence2", "sentence1_idx", "sentence2_idx"]
)
def slice_trigram(example):
    """If the target word share a trigram between sentences."""

    def get_ngrams(tokens, window=1):
        num_ngrams = len(tokens) - window + 1
        for i in range(num_ngrams):
            yield tokens[i : i + window]

    trigrams = []
    for sent, sent_idx in [
        (example.sentence1, example.sentence1_idx),
        (example.sentence2, example.sentence2_idx),
    ]:
        tokens = sent.split()
        trigrams.append(
            [
                " ".join(ngram).lower()
                for ngram in get_ngrams(tokens[sent_idx - 2 : sent_idx + 2], window=3)
                if len(ngram) == 3
            ]
        )
    return len(set(trigrams[0]).intersection(set(trigrams[1]))) > 0


@slicing_function(
    fields=["pos", "sentence1", "sentence2", "sentence1_idx", "sentence2_idx"]
)
def slice_mismatch_verb(example):
    """Is the target word a verb with different forms between sentences."""
    form1 = example.sentence1.split()[example.sentence1_idx]
    form2 = example.sentence2.split()[example.sentence2_idx]
    return (form1 != form2) and example.pos == "V"


@slicing_function(
    fields=["pos", "sentence1", "sentence2", "sentence1_idx", "sentence2_idx"]
)
def slice_mismatch_noun(example):
    """Is the target word a noun with different forms between sentences."""
    form1 = example.sentence1.split()[example.sentence1_idx]
    form2 = example.sentence2.split()[example.sentence2_idx]
    return (form1 != form2) and example.pos == "N"


slices = [
    slice_base,
    slice_verb,
    slice_noun,
    slice_trigram,
    slice_mismatch_verb,
    slice_mismatch_noun,
]

slice_func_dict = {slice.__name__: slice for slice in slices}
