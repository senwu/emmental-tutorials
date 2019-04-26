from task_config import LABEL_MAPPING
from emmental.task import EmmentalTask
from modules.bert_module import BertModule
from torch import nn
from modules.classification_module import ClassificationModule
from modules.regression_module import RegressionModule
from emmental.scorer import Scorer
import torch.nn.functional as F
from torch.nn import MSELoss

def mse_loss(immediate_ouput, Y):
    mse = MSELoss()
    return mse(immediate_ouput[-1][0].view(-1), Y.view(-1))


def ce_loss(immediate_ouput, Y):
    return F.cross_entropy(immediate_ouput[-1][0], Y.view(-1) - 1)


def output(immediate_ouput):
    return immediate_ouput[-1][0]


def get_gule_task(task_name, bert_model_name):

    task_cardinality = (
        len(LABEL_MAPPING[task_name].keys())
        if LABEL_MAPPING[task_name] is not None
        else 1
    )

    bert_output_dim = 768 if "uncased" in bert_model_name else 1024

    if task_name == "CoLA":
        task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "bert_module": BertModule(bert_model_name),
                    f"{task_name}_classification_module": ClassificationModule(
                        bert_output_dim, task_cardinality
                    ),
                }
            ),
            task_flow=[
                {
                    "module": "bert_module",
                    "inputs": [(0, "token_ids"), (0, "token_segments")],
                },
                {"module": f"{task_name}_classification_module", "inputs": [(1, 1)]},
            ],
            loss_func=ce_loss,
            output_func=output,
            scorer=Scorer(metrics=["matthews_correlation"]),
        )

    if task_name == "MNLI":
        task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "bert_module": BertModule(bert_model_name),
                    f"{task_name}_classification_module": ClassificationModule(
                        bert_output_dim, task_cardinality
                    ),
                }
            ),
            task_flow=[
                {
                    "module": "bert_module",
                    "inputs": [(0, "token_ids"), (0, "token_segments")],
                },
                {"module": f"{task_name}_classification_module", "inputs": [(1, 1)]},
            ],
            loss_func=ce_loss,
            output_func=output,
            scorer=Scorer(metrics=["accuracy"]),
        )

    if task_name == "MRPC":
        task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "bert_module": BertModule(bert_model_name),
                    f"{task_name}_classification_module": ClassificationModule(
                        bert_output_dim, task_cardinality
                    ),
                }
            ),
            task_flow=[
                {
                    "module": "bert_module",
                    "inputs": [(0, "token_ids"), (0, "token_segments")],
                },
                {"module": f"{task_name}_classification_module", "inputs": [(1, 1)]},
            ],
            loss_func=ce_loss,
            output_func=output,
            scorer=Scorer(metrics=["accuracy", "f1"]),
        )

    if task_name == "QNLI":
        task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "bert_module": BertModule(bert_model_name),
                    f"{task_name}_classification_module": ClassificationModule(
                        bert_output_dim, task_cardinality
                    ),
                }
            ),
            task_flow=[
                {
                    "module": "bert_module",
                    "inputs": [(0, "token_ids"), (0, "token_segments")],
                },
                {"module": f"{task_name}_classification_module", "inputs": [(1, 1)]},
            ],
            loss_func=ce_loss,
            output_func=output,
            scorer=Scorer(metrics=["accuracy"]),
        )

    if task_name == "QQP":
        task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "bert_module": BertModule(bert_model_name),
                    f"{task_name}_classification_module": ClassificationModule(
                        bert_output_dim, task_cardinality
                    ),
                }
            ),
            task_flow=[
                {
                    "module": "bert_module",
                    "inputs": [(0, "token_ids"), (0, "token_segments")],
                },
                {"module": f"{task_name}_classification_module", "inputs": [(1, 1)]},
            ],
            loss_func=ce_loss,
            output_func=output,
            scorer=Scorer(metrics=["accuracy", "f1"]),
        )

    if task_name == "RTE":
        task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "bert_module": BertModule(bert_model_name),
                    f"{task_name}_classification_module": ClassificationModule(
                        bert_output_dim, task_cardinality
                    ),
                }
            ),
            task_flow=[
                {
                    "module": "bert_module",
                    "inputs": [(0, "token_ids"), (0, "token_segments")],
                },
                {"module": f"{task_name}_classification_module", "inputs": [(1, 1)]},
            ],
            loss_func=ce_loss,
            output_func=output,
            scorer=Scorer(metrics=["accuracy"]),
        )

    if task_name == "SNLI":
        task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "bert_module": BertModule(bert_model_name),
                    f"{task_name}_classification_module": ClassificationModule(
                        bert_output_dim, task_cardinality
                    ),
                }
            ),
            task_flow=[
                {
                    "module": "bert_module",
                    "inputs": [(0, "token_ids"), (0, "token_segments")],
                },
                {"module": f"{task_name}_classification_module", "inputs": [(1, 1)]},
            ],
            loss_func=ce_loss,
            output_func=output,
            scorer=Scorer(metrics=["accuracy"]),
        )

    if task_name == "SST-2":
        task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "bert_module": BertModule(bert_model_name),
                    f"{task_name}_classification_module": ClassificationModule(
                        bert_output_dim, task_cardinality
                    ),
                }
            ),
            task_flow=[
                {
                    "module": "bert_module",
                    "inputs": [(0, "token_ids"), (0, "token_segments")],
                },
                {"module": f"{task_name}_classification_module", "inputs": [(1, 1)]},
            ],
            loss_func=ce_loss,
            output_func=output,
            scorer=Scorer(metrics=["accuracy"]),
        )

    if task_name == "STS-B":
        task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "bert_module": BertModule(bert_model_name),
                    f"{task_name}_regression_module": RegressionModule(bert_output_dim),
                }
            ),
            task_flow=[
                {
                    "module": "bert_module",
                    "inputs": [(0, "token_ids"), (0, "token_segments")],
                },
                {"module": f"{task_name}_regression_module", "inputs": [(1, 1)]},
            ],
            loss_func=mse_loss,
            output_func=output,
            scorer=Scorer(metrics=["pearson_spearman"]),
        )

    if task_name == "WNLI":
        task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "bert_module": BertModule(bert_model_name),
                    f"{task_name}_classification_module": ClassificationModule(
                        bert_output_dim, task_cardinality
                    ),
                }
            ),
            task_flow=[
                {
                    "module": "bert_module",
                    "inputs": [(0, "token_ids"), (0, "token_segments")],
                },
                {"module": f"{task_name}_classification_module", "inputs": [(1, 1)]},
            ],
            loss_func=ce_loss,
            output_func=output,
            scorer=Scorer(metrics=["accuracy"]),
        )

    return task
