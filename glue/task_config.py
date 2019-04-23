SPLIT_MAPPING = {
    "CoLA": {"train": "train.tsv", "dev": "dev.tsv", "test": "test.tsv"},
    "MNLI": {"train": "train.tsv", "dev": "dev_matched.tsv", "test": "test_matched.tsv"},
    "MNLI_matched": {"train": "train.tsv", "dev": "dev_matched.tsv", "test": "test_matched.tsv"},
    "MNLI_unmatched": {"train": "train.tsv", "dev": "dev_mismatched.tsv", "test": "test_mismatched.tsv"},
    "MRPC": {"train": "train.tsv", "dev": "dev.tsv", "test": "test.tsv"},
    "QNLI": {"train": "train.tsv", "dev": "dev.tsv", "test": "test.tsv"},
    "QQP": {"train": "train.tsv", "dev": "dev.tsv", "test": "test.tsv"},
    "RTE": {"train": "train.tsv", "dev": "dev.tsv", "test": "test.tsv"},
    "SNLI": {"train": "train.tsv", "dev": "dev.tsv", "test": "test.tsv"},
    "SST-2": {"train": "train.tsv", "dev": "dev.tsv", "test": "test.tsv"},
    "STS-B": {"train": "train.tsv", "dev": "dev.tsv", "test": "test.tsv"},
    "WNLI": {"train": "train.tsv", "dev": "dev.tsv", "test": "test.tsv"},
}

INDEX_MAPPING = {
    # each one contains three values, sentence 1 index, sentence 2 index, label index, -1 means abstain
    "CoLA": {"train": [3, -1, 1], "dev": [3, -1, 1], "test": [1, -1, -1]},
    "MNLI": {"train": [8, 9, 11], "dev": [8, 9, 15], "test": [8, 9, -1]},
    "MNLI_matched": {"train": [8, 9, 11], "dev": [8, 9, 15], "test": [8, 9, -1]},
    "MNLI_unmatched": {"train": [8, 9, 11], "dev": [8, 9, 15], "test": [8, 9, -1]},
    "MRPC": {"train": [3, 4, 0], "dev": [3, 4, 0], "test": [3, 4, 0]},
    "QNLI": {"train": [1, 2, 3], "dev": [1, 2, 3], "test": [1, 2, -1]},
    "QQP": {"train": [3, 4, 5], "dev": [3, 4, 5], "test": [1, 2, -1]},
    "RTE": {"train": [1, 2, 3], "dev": [1, 2, 3], "test": [1, 2, -1]},
    "SNLI": {"train": [6, 7, 1], "dev": [6, 7, 1], "test": [6, 7, 1]},
    "SST-2": {"train": [0, -1, 1], "dev": [0, -1, 1], "test": [1, -1, -1]},
    "STS-B": {"train": [7, 8, 9], "dev": [7, 8, 9], "test": [7, 8, -1]},
    "WNLI": {"train": [1, 2, 3], "dev": [1, 2, 3], "test": [1, 2, -1]},
}

SKIPPING_HEADER_MAPPING = {
    "CoLA": {"train": 0, "dev": 0, "test": 1},
    "MNLI": {"train": 1, "dev": 1, "test": 1},
    "MNLI_matched": {"train": 1, "dev": 1, "test": 1},
    "MNLI_unmatched": {"train": 1, "dev": 1, "test": 1},
    "MRPC": {"train": 1, "dev": 1, "test": 1},
    "QNLI": {"train": 1, "dev": 1, "test": 1},
    "QQP": {"train": 1, "dev": 1, "test": 1},
    "RTE": {"train": 1, "dev": 1, "test": 1},
    "SNLI": {"train": 1, "dev": 1, "test": 1},
    "SST-2": {"train": 1, "dev": 1, "test": 1},
    "STS-B": {"train": 1, "dev": 1, "test": 1},
    "WNLI": {"train": 1, "dev": 1, "test": 1},
}

LABEL_MAPPING = {
    "CoLA": {"1": 1, "0": 2},
    "MNLI": {"entailment": 1, "contradiction": 2, "neutral": 3},
    "MNLI_matched": {"entailment": 1, "contradiction": 2, "neutral": 3},
    "MNLI_unmatched": {"entailment": 1, "contradiction": 2, "neutral": 3},
    "MRPC": {"1": 1, "0": 2},
    "QNLI": {"entailment": 1, "not_entailment": 2},
    "QQP": {"1": 1, "0": 2},
    "RTE": {"entailment": 1, "not_entailment": 2},
    "SNLI": {"entailment": 1, "contradiction": 2, "neutral": 3},
    "SST-2": {"1": 1, "0": 2},
    "STS-B": None,
    "WNLI": {"1": 1, "0": 2},
}