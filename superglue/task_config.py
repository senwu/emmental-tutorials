SuperGLUE_TASK_NAMES = ["CB", "COPA", "MultiRC", "RTE", "WiC", "WSC"]

SuperGLUE_TASK_SPLIT_MAPPING = {
    "CB": {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"},
    "COPA": {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"},
    "MultiRC": {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"},
    "RTE": {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"},
    "WiC": {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"},
    #     "WSC": {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"},
    "WSC": {
        "train": "train.jsonl.retokenized.bert-large-cased",
        "val": "val.jsonl.retokenized.bert-large-cased",
        "test": "test.jsonl.retokenized.bert-large-cased",
    },
}

SuperGLUE_LABEL_MAPPING = {
    "CB": {"entailment": 1, "contradiction": 2, "neutral": 3},
    "COPA": {0: 1, 1: 2},
    "RTE": {"entailment": 1, "not_entailment": 2},
    "WiC": {True: 1, False: 2},
    "WSC": {True: 1, False: 2},
    "MultiRC": {True: 1, False: 2},
}

SuperGLUE_TASK_METRIC_MAPPING = {
    "CB": ["accuracy"],
    "COPA": ["accuracy"],
    "MultiRC": ["f1"],
    "RTE": ["accuracy"],
    "WiC": ["accuracy"],
    "WSC": ["accuracy"],
}
