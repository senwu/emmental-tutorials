from functools import partial

import torch.nn.functional as F
from torch import nn
from torch.nn import MSELoss

from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from modules.bert_module import BertModule
from modules.classification_module import ClassificationModule
from modules.regression_module import RegressionModule
from task_config import LABEL_MAPPING, METRIC_MAPPING


def ce_loss(task_name, immediate_ouput_dict, Y, active):
    module_name = f"{task_name}_pred_head"
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0][active], (Y.view(-1) - 1)[active]
    )


def mse_loss(task_name, immediate_ouput_dict, Y, active):
    mse = MSELoss()
    module_name = f"{task_name}_pred_head"
    return mse(
        immediate_ouput_dict[module_name][0][active].view(-1), Y[active].view(-1)
    )


def output(task_name, immediate_ouput_dict):
    module_name = f"{task_name}_pred_head"
    return immediate_ouput_dict[module_name][0]


def get_gule_task(task_names, bert_model_name):

    tasks = dict()

    bert_module = BertModule(bert_model_name)
    bert_output_dim = 768 if "base" in bert_model_name else 1024

    for task_name in task_names:
        task_cardinality = (
            len(LABEL_MAPPING[task_name].keys())
            if LABEL_MAPPING[task_name] is not None
            else 1
        )

        metrics = METRIC_MAPPING[task_name]

        if task_name == "STS-B":
            loss_fn = partial(mse_loss, task_name)
        else:
            loss_fn = partial(ce_loss, task_name)

        task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "bert_module": bert_module,
                    f"{task_name}_pred_head": nn.Linear(
                        bert_output_dim, task_cardinality
                    ),
                }
            ),
            task_flow=[
                {
                    "name": "input",
                    "module": "bert_module",
                    "inputs": [("_input_", "token_ids"), ("_input_", "token_segments")],
                },
                {
                    "name": f"{task_name}_pred_head",
                    "module": f"{task_name}_pred_head",
                    "inputs": [("input", 1)],
                },
            ],
            loss_func=loss_fn,
            output_func=partial(output, task_name),
            scorer=Scorer(metrics=metrics),
        )

        tasks[task_name] = task

    return tasks
