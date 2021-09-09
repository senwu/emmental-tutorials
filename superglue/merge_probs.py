"""
Make a single prediction file by combining multiple probabilities files.

NOTE: This will not work for MultiRC
"""

import os
import subprocess
from collections import defaultdict

import click
import jsonlines
import numpy as np
from task_config import SuperGLUE_LABEL_INVERSE


@click.command()
@click.option("--task_name", required=True)
@click.option("--root", required=True)
def merge_probs(root, task_name):
    INFILES = subprocess.check_output(
        f'find {root} -type f -name "*probs.jsonl"', shell=True
    ).split()
    OUTFILE = os.path.join(root, f"{task_name}.jsonl")

    # Print evaluation scores
    subprocess.call(
        r"find " + root + r' -type f -name "best_metrics.txt" -exec cat {} \;',
        shell=True,
    )

    preds_dict = defaultdict(list)
    for probs_file in INFILES:
        with jsonlines.open(probs_file, "r") as reader:
            for line in reader:
                probs = np.array([float(p) for p in line["probs"][1:-1].split()])
                preds_dict[line["idx"]].append(probs)

    preds_formatted = []
    for idx, probs_list in preds_dict.items():
        pred = np.argmax(np.array(probs_list).mean(axis=0)) + 1
        label = str(SuperGLUE_LABEL_INVERSE[task_name][pred]).lower()
        preds_formatted.append({"idx": idx, "label": label})

    with jsonlines.open(OUTFILE, "w") as writer:
        writer.write_all(preds_formatted)


if __name__ == "__main__":
    merge_probs()
