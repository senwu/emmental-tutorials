import logging
import os
import re
import shutil
from collections import defaultdict

import click
import jsonlines
from dataloaders import get_dataloaders
from models.cb import build_model as build_model_cb
from models.copa import build_model as build_model_copa
from models.multirc import build_model as build_model_multirc
from models.rte import build_model as build_model_rte
from models.wic import build_model as build_model_wic
from models.wsc import build_model as build_model_wsc
from task_config import SuperGLUE_LABEL_INVERSE

import emmental
from emmental.model import EmmentalModel

logger = logging.getLogger(__name__)

build_model = {
    "CB": build_model_cb,
    "COPA": build_model_copa,
    "MultiRC": build_model_multirc,
    "RTE": build_model_rte,
    "WiC": build_model_wic,
    "WSC": build_model_wsc,
}

TASKS = ["CB", "COPA", "MultiRC", "RTE", "WiC", "WSC"]


def format_multirc_preds(dataloader, preds):
    # Make prediction map
    paragraphs = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for idx, pred in enumerate(zip(preds)):
        pid = dataloader.dataset.X_dict["pids"][idx]
        qid = dataloader.dataset.X_dict["qids"][idx]
        aid = dataloader.dataset.X_dict["aids"][idx]
        paragraphs[pid][qid][aid] = pred
    # Moves indices from keys to sibling fields
    preds_formatted = []
    for pid, questions in paragraphs.items():
        p_dict = {"idx": pid, "paragraph": {"questions": []}}
        for qid, question in questions.items():
            q_dict = {"idx": qid, "answers": []}
            for aid, pred in question.items():
                label = str(SuperGLUE_LABEL_INVERSE["MultiRC"][pred]).lower()
                answer = {"idx": aid, "label": str(label)}
                q_dict["answers"].append(answer)
            p_dict["paragraph"]["questions"].append(q_dict)
        preds_formatted.append(p_dict)
    return preds_formatted


def extract_from_cmd(path):
    log_dir = os.path.dirname(path)
    with open(os.path.join(log_dir, "cmd.txt")) as f:
        cmd = f.read()

    bert_model_match = re.search(r"--bert_model[ =](bert-\S+)", cmd)
    if bert_model_match:
        bert_model_name = bert_model_match.group(1)
    else:
        bert_model_name = "bert-large-cased"

    max_seq_match = re.search(r"--max_sequence_length[ =](\d+)", cmd)
    if max_seq_match:
        max_seq_len = int(max_seq_match.group(1))
    else:
        max_seq_len = 256
    return bert_model_name, max_seq_len


def predict_and_write(task_name, path, data_dir, submit_subdir, batch_size):
    bert_model_name, max_seq_len = extract_from_cmd(path)
    msg = (
        f"Using {bert_model_name} and max_sequence_len={max_seq_len} for task "
        f"{task_name}"
    )
    logger.info(msg)

    # Build model
    task = build_model[task_name](bert_model_name)
    model = EmmentalModel(name=f"SuperGLUE_{task_name}", tasks=[task])
    try:
        model.load(path)
    except UnboundLocalError:
        msg = (
            "Failed to load state dict; confirm that your model was saved with "
            "a command such as 'torch.save(model.state_dict(), PATH)'"
        )
        logging.error(msg)
        raise

    # Build dataloaders
    dataloaders = get_dataloaders(
        data_dir,
        task_name=task_name,
        splits=["val", "test"],  # TODO: replace with ['split'] and update below
        max_data_samples=None,
        max_sequence_length=max_seq_len,
        tokenizer_name=bert_model_name,
        batch_size=batch_size,
        uid="uids",
    )
    # TEMP: Sanity check val performance
    logger.info(f"Valid score: {model.score(dataloaders[0])}")
    # TEMP

    filename = f"{task_name}.jsonl"
    filepath = os.path.join(submit_subdir, filename)
    make_submission_file(model, dataloaders[-1], task_name, filepath)


def make_submission_file(model, dataloader, task_name, filepath):
    output = model.predict(dataloader, return_preds=True)
    preds = output["preds"][task_name]
    probs = output["probs"][task_name]

    probs_filepath = os.path.join(
        os.path.dirname(filepath),
        "probs",
        os.path.basename(filepath).replace(".jsonl", "_probs.jsonl"),
    )
    if not os.path.exists(os.path.dirname(probs_filepath)):
        os.mkdir(os.path.dirname(probs_filepath))

    if task_name == "MultiRC":
        preds_formatted = format_multirc_preds(dataloader, preds)
    else:
        probs_formatted = []
        preds_formatted = []
        for idx, y in enumerate(preds):
            label = str(SuperGLUE_LABEL_INVERSE[task_name][y]).lower()
            preds_formatted.append({"idx": idx, "label": label})
            probs_formatted.append({"idx": idx, "probs": str(probs[idx])})
        logger.info(f"Writing probabilities to {probs_filepath}")
        with jsonlines.open(probs_filepath, mode="w") as writer:
            writer.write_all(probs_formatted)

    logger.info(f"Writing predictions to {filepath}")
    with jsonlines.open(filepath, mode="w") as writer:
        writer.write_all(preds_formatted)


@click.command()
@click.option("--CB", help="Path to CB model")
@click.option("--COPA", help="Path to COPA model")
@click.option("--MultiRC", help="Path to MultiRC model")
@click.option("--RTE", help="Path to RTE model")
@click.option("--WiC", help="Path to WiC model")
@click.option("--WSC", help="Path to WSC model")
@click.option("--split", default="test", type=click.Choice(["train", "val", "test"]))
@click.option("--batch_size", default=4, type=int)
@click.option("--data_dir", help="Path to data")
@click.option("--submit_dir", default="submissions")
@click.argument("name")
def make_submission(
    name, split, data_dir, submit_dir, batch_size, cb, copa, multirc, rte, wic, wsc
):
    submit_subdir = os.path.join(submit_dir, name)
    if not os.path.exists(submit_subdir):
        os.makedirs(submit_subdir)

    for task_name, path in zip(TASKS, [cb, copa, multirc, rte, wic, wsc]):
        if not path:
            continue

        predictions_path = os.path.join(os.path.dirname(path), f"{task_name}.jsonl")
        if os.path.exists(predictions_path):
            logger.info(f"{task_name}: Found predictions file.")
            shutil.copy(predictions_path, submit_subdir)
        else:
            logger.info(
                f"{task_name}: No predictions file found. Generating predictions..."
            )
            predict_and_write(task_name, path, data_dir, submit_subdir, batch_size)

    shutil.copy(os.path.join(submit_dir, "AX.jsonl"), submit_subdir)

    os.chdir(submit_subdir)
    os.system("zip -r submission.zip *.jsonl")
    os.chdir("..")


if __name__ == "__main__":
    emmental.init()
    make_submission()
