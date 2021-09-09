"""Converts SemCor dataset from xml word-per-line to jsonl sentence pairs."""

import logging
import random
import sys
from collections import defaultdict, namedtuple
from itertools import combinations

import jsonlines
import numpy as np
import xmltodict

sys.path.append("..")  # Adds higher directory to python modules path.


logger = logging.getLogger(__name__)

Candidate = namedtuple("Candidate", ["word", "pos", "sense", "sentence"])


def parse(in_paths, out_path, max_per_word=5):
    assert out_path.endswith(".jsonl")

    # Files to sentences
    sentences = []
    sentence = []
    for file_path in in_paths:
        logger.info(f"Loading data from {file_path}.")
        with open(file_path) as f:
            doc = xmltodict.parse(f.read())
        for word in doc["SimpleWsdDoc"]["word"]:
            if word["@break_level"] in ["SENTENCE_BREAK", "PARAGRAPH_BREAK"]:
                sentences.append(sentence)
                sentence = [word]
            else:
                sentence.append(word)
        sentences.append(sentence)

    # Sentences to groups
    sentence_groups = defaultdict(list)
    for sentence in sentences:
        sentence_text = " ".join([word["@text"] for word in sentence])
        for word in sentence:
            if word.get("@sense"):
                cand = Candidate(
                    word["@text"], word["@pos"], word["@sense"], sentence_text
                )
                sentence_groups[(word["@lemma"], word["@pos"])].append(cand)

    # Groups to pairs
    idx = 0
    examples = []
    for group, sentences in sentence_groups.items():
        if group[1] not in ["VERB", "NOUN"]:
            continue
        pairs = [p for p in combinations(sentences, 2)]
        random.shuffle(pairs)
        for s1, s2 in pairs[:max_per_word]:
            example = {
                "label": s1.sense == s2.sense,
                "word": s1.word,
                "pos": s1.pos[0],
                "sentence1": s1.sentence,
                "sentence2": s2.sentence,
                "sentence1_idx": s1.sentence.split().index(s1.word),
                "sentence2_idx": s2.sentence.split().index(s2.word),
            }
            examples.append(example)
            idx += 1

    # Write to file
    labels = np.array([e["label"] for e in examples])
    print(f"Writing new examples to {out_path}")
    print(f"Total examples: {len(examples)}")
    print(f"Unique words: {len(set([e['word'] for e in examples]))}")
    print(
        f"Class balance: {sum(labels == 1)}/{len(labels)} "
        f"({sum(labels == 1)/len(labels):.2f}) positives"
    )
    with jsonlines.open(out_path, mode="w") as writer:
        writer.write_all(examples)
    return examples


if __name__ == "__main__":
    in_paths = [
        "/dfs/scratch1/bradenjh/word_sense_disambigation_corpora/semcor/br-a01.xml"
    ]
    out_path = "/dfs/scratch1/bradenjh/temp/br-a01.jsonl"
    examples = parse(in_paths, out_path)
