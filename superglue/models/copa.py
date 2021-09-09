import sys
from functools import partial

from modules.bert_module import BertLastCLSModule, BertModule
from modules.multiple_choice_module import MultipleChoiceModule
from task_config import SuperGLUE_TASK_METRIC_MAPPING
from torch import nn

from emmental.scorer import Scorer
from emmental.task import EmmentalTask

from . import utils

sys.path.append("..")  # Adds higher directory to python modules path.


TASK_NAME = "COPA"
NUM_CHOICES = 2


def build_model(bert_model_name, last_hidden_dropout_prob=0.0):

    bert_module = BertModule(bert_model_name)
    bert_output_dim = 768 if "base" in bert_model_name else 1024

    metrics = (
        SuperGLUE_TASK_METRIC_MAPPING[TASK_NAME]
        if TASK_NAME in SuperGLUE_TASK_METRIC_MAPPING
        else []
    )

    customize_metric_funcs = {}

    loss_fn = partial(
        utils.ce_loss_multiple_choice, f"{TASK_NAME}_pred_head", NUM_CHOICES
    )
    output_fn = partial(
        utils.output_multiple_choice, f"{TASK_NAME}_pred_head", NUM_CHOICES
    )

    task = EmmentalTask(
        name=TASK_NAME,
        module_pool=nn.ModuleDict(
            {
                f"{TASK_NAME}_multiple_choice_module": MultipleChoiceModule(
                    NUM_CHOICES
                ),
                "bert_module": bert_module,
                f"{TASK_NAME}_feature": BertLastCLSModule(
                    dropout_prob=last_hidden_dropout_prob
                ),
                f"{TASK_NAME}_pred_head": nn.Linear(bert_output_dim, 1),
            }
        ),
        task_flow=[
            {
                "name": f"{TASK_NAME}_multiple_choice_module",
                "module": f"{TASK_NAME}_multiple_choice_module",
                "inputs": [],
            },
            {
                "name": f"{TASK_NAME}_bert_module",
                "module": "bert_module",
                "inputs": [
                    (f"{TASK_NAME}_multiple_choice_module", 0),
                    (f"{TASK_NAME}_multiple_choice_module", 1),
                    (f"{TASK_NAME}_multiple_choice_module", 2),
                ],
            },
            {
                "name": f"{TASK_NAME}_feature",
                "module": f"{TASK_NAME}_feature",
                "inputs": [(f"{TASK_NAME}_bert_module", 0)],
            },
            {
                "name": f"{TASK_NAME}_pred_head",
                "module": f"{TASK_NAME}_pred_head",
                "inputs": [(f"{TASK_NAME}_feature", 0)],
            },
        ],
        loss_func=loss_fn,
        output_func=output_fn,
        scorer=Scorer(metrics=metrics, customize_metric_funcs=customize_metric_funcs),
    )

    return task
