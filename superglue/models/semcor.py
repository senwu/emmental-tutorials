import sys
from functools import partial

from modules.bert_module import BertContactLastCLSWithTwoTokensModule, BertModule
from task_config import SuperGLUE_LABEL_MAPPING, SuperGLUE_TASK_METRIC_MAPPING
from torch import nn

from emmental.scorer import Scorer
from emmental.task import EmmentalTask

from . import utils

sys.path.append("..")  # Adds higher directory to python modules path.


TASK_NAME = "SemCor"


def build_model(bert_model_name, last_hidden_dropout_prob=None):
    if last_hidden_dropout_prob:
        raise NotImplementedError(f"TODO: last_hidden_dropout_prob for {TASK_NAME}")

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

    customize_metric_funcs = {}

    loss_fn = partial(utils.ce_loss, f"{TASK_NAME}_pred_head")
    output_fn = partial(utils.output, f"{TASK_NAME}_pred_head")

    task = EmmentalTask(
        name=TASK_NAME,
        module_pool=nn.ModuleDict(
            {
                "bert_module": bert_module,
                f"{TASK_NAME}_feature": BertContactLastCLSWithTwoTokensModule(),
                f"{TASK_NAME}_pred_head": nn.Linear(
                    bert_output_dim * 3, task_cardinality
                ),
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
                "name": f"{TASK_NAME}_feature",
                "module": f"{TASK_NAME}_feature",
                "inputs": [
                    (f"{TASK_NAME}_bert_module", 0),
                    ("_input_", "token1_idx"),
                    ("_input_", "token2_idx"),
                ],
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
