import logging
import os

import click
import jsonlines

from dataloaders import get_dataloaders
import emmental
from emmental.model import EmmentalModel
from models.wic import build_model as build_model_wic
from models.cb import build_model as build_model_cb

build_model = {
    "CB": build_model_cb,
    # "COPA": build_model_copa,
    # "MultiRC": build_model_multirc,
    # "RTE": build_model_rte,
    "WiC": build_model_wic,
    # "WSC": build_model_wsc,
}

def preds_to_ent_contr(preds):
    """Converts predictions (1s and 2s) into strings (entailment/contradiction)"""
    return ["entailment" if x == 1 else "contradiction" for x in preds]

def preds_to_01(preds):
    """Converts predictions (1s and 2s) into binary labels (0/1)"""
    return [1 if x == 1 else 0 for x in preds]

def preds_to_ent_notent(preds):
    """Converts predictions (1s and 2s) into strings (entailment/not_entailment)"""
    return ["entailment" if x == 1 else "not_entailment" for x in preds]

def preds_to_true_false(preds):
    """Converts predictions (1s and 2s) into strings (true/false)"""
    return ["true" if x == 1 else "false" for x in preds]



preds_to_output = {
    "CB": preds_to_ent_contr,
    "COPA": preds_to_01,
    "MultiRC": None, # TBD
    "RTE": preds_to_ent_notent,
    "WiC": preds_to_true_false,
    "WSC": preds_to_true_false,
}

CB_MODEL = "/path/to/model"
COPA_MODEL = "/path/to/model"
MultiRC_MODEL = "/path/to/model"
RTE_MODEL = "/path/to/model"
WiC_MODEL = "/path/to/model"
WSC_MODEL = "/path/to/model"

TASKS = ["CB", "COPA", "MultiRC", "RTE", "WiC", "WSC"]
BERT_MODEL_NAME = "bert-large-cased"

@click.command()
@click.option('--CB', default=CB_MODEL, help="Path to CB model")
@click.option('--COPA', default=COPA_MODEL, help="Path to COPA model")
@click.option('--MultiRC', default=MultiRC_MODEL, help="Path to MultiRC model")
@click.option('--RTE', default=RTE_MODEL, help="Path to RTE model")
@click.option('--WiC', default=WiC_MODEL, help="Path to WiC model")
@click.option('--WSC', default=WSC_MODEL, help="Path to WSC model")
@click.option('--split', default="test", type=click.Choice(["train", "val", "test"]))
@click.option('--data-dir', default=os.environ["SUPERGLUEDATA"])
@click.argument('name')
def make_submission(name, split, data_dir, cb, copa, multirc, rte, wic, wsc):
    submit_dir = f"submissions/{name}/"
    if not os.path.exists(os.path.dirname(submit_dir)):
        os.makedirs(os.path.dirname(submit_dir))
    
    for task_name, path in zip(TASKS, [cb, copa, multirc, rte, wic, wsc]):
        if task_name not in ["CB", "WiC"]:
            continue
        task = build_model[task_name](BERT_MODEL_NAME)
        model = EmmentalModel(name=f"SuperGLUE_{task_name}", tasks=[task])
        try:
            model.load(path)
        except UnboundLocalError as e:
            msg = ("Failed to load state dict; confirm that your model was saved with "
                   "a command such as 'torch.save(model.state_dict(), PATH)'")
            logging.error(msg)
            raise
        dataloaders = get_dataloaders(
            data_dir,
            task_name=task_name,
            splits=["val", "test"], # TODO: replace with ['split'] and update below
            max_data_samples=None,
            # max_sequence_length=128,
            tokenizer_name=BERT_MODEL_NAME,
            batch_size=16,
            uid="uids",
        )
        # TEMP: Sanity check val performance
        logging.info(model.score(dataloaders[0]))
        # TEMP
        
        preds = model.predict(dataloaders[1], return_preds=True)["preds"][task_name]
        logging.info(preds)

        preds_formatted = []
        for idx, y in enumerate(preds_to_output[task_name](preds)):
            preds_formatted.append({"idx": idx, "label": y})

        filename = f'{task_name}.jsonl'
        filepath = os.path.join(submit_dir, filename)
        logging.info(f"Writing predictions to {filepath}")
        with jsonlines.open(filepath, mode='w') as writer:
            writer.write_all(preds_formatted)

    os.chdir(submit_dir)
    os.system("zip -r submission.zip *.jsonl")
    os.chdir("..")


if __name__ == '__main__':
    emmental.init()
    make_submission()