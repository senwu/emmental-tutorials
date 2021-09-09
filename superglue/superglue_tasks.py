import collections
from functools import partial

import torch.nn.functional as F
from modules.bert_module import BertLastCLSModule, BertModule
from task_config import SuperGLUE_LABEL_MAPPING, SuperGLUE_TASK_METRIC_MAPPING
from torch import nn

from emmental.metrics.fbeta import f1_scorer
from emmental.scorer import Scorer
from emmental.task import EmmentalTask


def ce_loss(module_name, immediate_ouput_dict, Y, active):
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0][active], (Y.view(-1) - 1)[active]
    )


def output(module_name, immediate_ouput_dict):
    return F.softmax(immediate_ouput_dict[module_name][0], dim=1)


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


def get_superglue_task(task_names, bert_model_name):

    tasks = dict()

    bert_module = BertModule(bert_model_name)
    bert_output_dim = 768 if "base" in bert_model_name else 1024

    for task_name in task_names:
        task_cardinality = (
            len(SuperGLUE_LABEL_MAPPING[task_name].keys())
            if SuperGLUE_LABEL_MAPPING[task_name] is not None
            else 1
        )

        metrics = (
            SuperGLUE_TASK_METRIC_MAPPING[task_name]
            if task_name in SuperGLUE_TASK_METRIC_MAPPING
            else []
        )
        customize_metric_funcs = (
            {"em": em, "em_f1": em_f1} if task_name == "MultiRC" else {}
        )

        loss_fn = partial(ce_loss, f"{task_name}_pred_head")
        output_fn = partial(output, f"{task_name}_pred_head")

        if task_name == "MultiRC":
            task = EmmentalTask(
                name=task_name,
                module_pool=nn.ModuleDict(
                    {
                        "bert_module": bert_module,
                        "bert_last_CLS": BertLastCLSModule(),
                        f"{task_name}_pred_head": nn.Linear(
                            bert_output_dim, task_cardinality
                        ),
                    }
                ),
                task_flow=[
                    {
                        "name": f"{task_name}_bert_module",
                        "module": "bert_module",
                        "inputs": [("_input_", "token_ids")],
                    },
                    {
                        "name": f"{task_name}_bert_last_CLS",
                        "module": "bert_last_CLS",
                        "inputs": [(f"{task_name}_bert_module", 0)],
                    },
                    {
                        "name": f"{task_name}_pred_head",
                        "module": f"{task_name}_pred_head",
                        "inputs": [(f"{task_name}_bert_last_CLS", 0)],
                    },
                ],
                loss_func=loss_fn,
                output_func=output_fn,
                scorer=Scorer(
                    metrics=metrics, customize_metric_funcs=customize_metric_funcs
                ),
            )

            tasks[task_name] = task

    return tasks
