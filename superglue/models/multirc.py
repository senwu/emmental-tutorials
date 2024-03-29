import collections
import sys
from functools import partial

from modules.bert_module import BertLastCLSModule, BertModule
from task_config import SuperGLUE_LABEL_MAPPING, SuperGLUE_TASK_METRIC_MAPPING
from torch import nn

from emmental.metrics.fbeta import f1_scorer
from emmental.scorer import Scorer
from emmental.task import EmmentalTask

from . import utils

sys.path.append("..")  # Adds higher directory to python modules path.


TASK_NAME = "MultiRC"


# customize_metric_funcs #################


def em(golds, probs, preds, uids):
    gt_pds = collections.defaultdict(list)

    for gold, pred, uid in zip(golds, preds, uids):
        qid = "%%".join(uid.split("%%")[:2])
        gt_pds[qid].append((gold, pred))

    cnt, tot = 0, 0
    for gt_pd in gt_pds.values():
        tot += 1
        gt, pd = list(zip(*gt_pd))
        if gt == pd:
            cnt += 1

    return cnt / tot


def em_f1(golds, probs, preds, uids):
    f1 = f1_scorer(golds, probs, preds, uids)
    exact = em(golds, probs, preds, uids)

    return (exact + f1["f1"]) / 2


#########################################


def build_model(bert_model_name, last_hidden_dropout_prob=0.0):

    bert_module = BertModule(bert_model_name)
    bert_output_dim = 768 if "base" in bert_model_name else 1024

    task_cardinality = (
        len(SuperGLUE_LABEL_MAPPING[TASK_NAME].keys())
        if SuperGLUE_LABEL_MAPPING[TASK_NAME] is not None
        else 1
    )

    metrics = (
        SuperGLUE_TASK_METRIC_MAPPING[TASK_NAME]
        if TASK_NAME in SuperGLUE_TASK_METRIC_MAPPING
        else []
    )

    customize_metric_funcs = {"em": em, "em_f1": em_f1}

    loss_fn = partial(utils.ce_loss, f"{TASK_NAME}_pred_head")
    output_fn = partial(utils.output, f"{TASK_NAME}_pred_head")

    task = EmmentalTask(
        name=TASK_NAME,
        module_pool=nn.ModuleDict(
            {
                "bert_module": bert_module,
                "bert_last_CLS": BertLastCLSModule(
                    dropout_prob=last_hidden_dropout_prob
                ),
                f"{TASK_NAME}_pred_head": nn.Linear(bert_output_dim, task_cardinality),
            }
        ),
        task_flow=[
            {
                "name": f"{TASK_NAME}_bert_module",
                "module": "bert_module",
                "inputs": [
                    ("_input_", "token_ids"),
                    ("_input_", "token_segments"),
                    ("_input_", "token_masks"),
                ],
            },
            {
                "name": f"{TASK_NAME}_bert_last_CLS",
                "module": "bert_last_CLS",
                "inputs": [(f"{TASK_NAME}_bert_module", 0)],
            },
            {
                "name": f"{TASK_NAME}_pred_head",
                "module": f"{TASK_NAME}_pred_head",
                "inputs": [(f"{TASK_NAME}_bert_last_CLS", 0)],
            },
        ],
        loss_func=loss_fn,
        output_func=output_fn,
        scorer=Scorer(metrics=metrics, customize_metric_funcs=customize_metric_funcs),
    )

    return task
