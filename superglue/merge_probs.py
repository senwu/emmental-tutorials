"""
Make a single prediction file by combining multiple probabilities files

NOTE: This will not work for MultiRC
"""

from collections import defaultdict
import os

import jsonlines
import numpy as np

from task_config import SuperGLUE_LABEL_INVERSE

TASK_NAME = "CB"

ROOT = "/dfs/scratch1/bradenjh/emmental-tutorials/superglue/logs/CB/test_cv_6/2019_06_12"
IN_FILES = [
    os.path.join(ROOT, "12_17_26/probs/CB_probs.jsonl"),
    os.path.join(ROOT, "12_19_49/probs/CB_probs.jsonl"),
    os.path.join(ROOT, "12_21_46/probs/CB_probs.jsonl"),
    os.path.join(ROOT, "12_23_35/probs/CB_probs.jsonl"),
    os.path.join(ROOT, "12_25_30/probs/CB_probs.jsonl"),
]

OUTFILE = os.path.join(ROOT, f"{TASK_NAME}.jsonl")

preds_dict = defaultdict(list)
for probs_file in IN_FILES:
    with jsonlines.open(probs_file, 'r') as reader:
        for line in reader:
            probs = np.array([float(p) for p in line["probs"][1:-1].split()])
            preds_dict[line["idx"]].append(probs)

preds_formatted = []
for idx, probs_list in preds_dict.items():
    pred = np.argmax(np.array(probs_list).mean(axis=0)) + 1
    label = str(SuperGLUE_LABEL_INVERSE[TASK_NAME][pred]).lower()
    preds_formatted.append({"idx": idx, "label": label})

with jsonlines.open(OUTFILE, 'w') as writer:
    writer.write_all(preds_formatted)

