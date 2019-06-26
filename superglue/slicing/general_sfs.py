import logging

from slicing.slicing_function import slicing_function

logger = logging.getLogger(__name__)


@slicing_function(fields=["sentence1", "sentence2"])
def slice_temporal_preposition(example):
    temporal_prepositions = ["after", "before", "past"]
    both_sentences = example.sentence1 + example.sentence2
    return any([p in both_sentences for p in temporal_prepositions])


@slicing_function(fields=["sentence1", "sentence2"])
def slice_possessive_preposition(example):
    possessive_prepositions = ["inside of", "with", "within"]
    both_sentences = example.sentence1 + example.sentence2
    return any([p in both_sentences for p in possessive_prepositions])


@slicing_function(fields=["sentence1", "sentence2"])
def slice_is_comparative(example):
    comparative_words = ["more", "less", "better", "worse", "bigger", "smaller"]
    both_sentences = example.sentence1 + example.sentence2
    return any([p in both_sentences for p in comparative_words])


@slicing_function(fields=["sentence1", "sentence2"])
def slice_is_quantification(example):
    quantification_words = ["all", "some", "none"]
    both_sentences = example.sentence1 + example.sentence2
    return any([p in both_sentences for p in quantification_words])


@slicing_function(fields=["sentence2"])
def slice_short_hypothesis(example, thresh=5):
    return len(example.sentence2.split()) < thresh


@slicing_function(fields=["sentence2"])
def slice_long_hypothesis(example, thresh=15):
    return len(example.sentence2.split()) > thresh


@slicing_function(fields=["sentence1"])
def slice_short_premise(example, thresh=10):
    return len(example.sentence1.split()) < thresh


@slicing_function(fields=["sentence1"])
def slice_long_premise(example, thresh=100):
    return len(example.sentence1.split()) > thresh


@slicing_function(fields=["sentence1", "sentence2"])
def slice_where(example):
    sentences = example.sentence1 + example.sentence2
    return "where" in sentences


@slicing_function(fields=["sentence1", "sentence2"])
def slice_who(example):
    sentences = example.sentence1 + example.sentence2
    return "who" in sentences


@slicing_function(fields=["sentence1", "sentence2"])
def slice_what(example):
    sentences = example.sentence1 + example.sentence2
    return "what" in sentences


@slicing_function(fields=["sentence1", "sentence2"])
def slice_when(example):
    sentences = example.sentence1 + example.sentence2
    return "when" in sentences


@slicing_function(fields=["sentence1", "sentence2"])
def slice_and(example):
    sentences = example.sentence1 + example.sentence2
    return "and" in sentences


@slicing_function(fields=["sentence1", "sentence2"])
def slice_but(example):
    sentences = example.sentence1 + example.sentence2
    return "but" in sentences


@slicing_function(fields=["sentence1", "sentence2"])
def slice_or(example):
    sentences = example.sentence1 + example.sentence2
    return "or" in sentences


@slicing_function(fields=["sentence1", "sentence2"])
def slice_multiple_articles(example):
    sentences = example.sentence1 + example.sentence2
    multiple_indefinite = (
        sum([int(x == "a") for x in sentences.split()]) > 1
        or sum([int(x == "an") for x in sentences.split()]) > 1
    )
    multiple_definite = sum([int(x == "the") for x in sentences.split()]) > 1
    return multiple_indefinite or multiple_definite


slices = [
    slice_temporal_preposition,
    slice_possessive_preposition,
    slice_is_comparative,
    slice_is_quantification,
    slice_short_hypothesis,
    slice_long_hypothesis,
    slice_short_premise,
    slice_long_premise,
    slice_where,
    slice_who,
    slice_what,
    slice_when,
    slice_and,
    slice_or,
    slice_but,
    slice_multiple_articles,
]

slice_func_dict = {slice.__name__: slice for slice in slices}
