import logging
import sys
from functools import partial

import torch
from torch import nn
from emmental.scorer import Scorer

from emmental import Meta
from emmental.task import EmmentalTask
from models import utils

from . import master_module

from . import \
    CB_slices, WiC_slices #; COPA_slices,; MultiRC_slices,; RTE_slices,; WSC_slices,

sys.path.append("..")  # Adds higher directory to python modules path.


slice_func_dict = {
    "CB": CB_slices.slice_func_dict,
    # "COPA": COPA_slices.slice_func_dict,
    # "MultiRC": MultiRC_slices.slice_func_dict,
    # "RTE": RTE_slices.slice_func_dict,
    "WiC": WiC_slices.slice_func_dict,
    # "WSC": WSC_slices.slice_func_dict,
}

logger = logging.getLogger(__name__)


def add_slice_labels(task_name, dataloaders, slice_func_dict):
    for dataloader in dataloaders:
        for slice_name, slice_func in slice_func_dict.items():
            ind, pred = slice_func(dataloader.dataset)
            dataloader.dataset.Y_dict.update(
                {
                    f"{task_name}_slice_ind_{slice_name}": ind,
                    f"{task_name}_slice_pred_{slice_name}": pred,
                }
            )
            dataloader.task_to_label_dict.update(
                {
                    f"{task_name}_slice_ind_{slice_name}": f"{task_name}_slice_ind_{slice_name}",
                    f"{task_name}_slice_pred_{slice_name}": f"{task_name}_slice_pred_{slice_name}",
                }
            )
        main_label = dataloader.task_to_label_dict[task_name]
        del dataloader.task_to_label_dict[task_name]
        dataloader.task_to_label_dict.update({task_name: main_label})

        msg = (
            f"Loaded slice labels for task {task_name}, slice {slice_name}, "
            f"split {dataloader.split}."
        )
        logger.info(msg)

    return dataloaders


def add_slice_tasks(task_name, base_task, slice_func_dict, hidden_dim=1024):

    tasks = []

    # base task info
    base_module_pool = base_task.module_pool
    base_task_flow = base_task.task_flow
    base_scorer = base_task.scorer

    # sanity check the model
    assert f"{task_name}_feature" in [
        i["name"] for i in base_task_flow
    ], f"{task_name}_feature should in the task module_pool"

    assert (
        isinstance(base_module_pool[f"{task_name}_pred_head"], nn.Linear) == True
    ), f"{task_name}_pred_head should be a nn.Linear layer"

    # extract last layer info
    last_linear_layer_size = (
        (
            base_module_pool[f"{task_name}_pred_head"].module.in_features,
            base_module_pool[f"{task_name}_pred_head"].module.out_features,
        )
        if Meta.config["model_config"]["dataparallel"]
        else (
            base_module_pool[f"{task_name}_pred_head"].in_features,
            base_module_pool[f"{task_name}_pred_head"].out_features,
        )
    )

    # remove the origin head
    del base_module_pool[f"{task_name}_pred_head"]
    for idx, i in enumerate(base_task_flow):
        if i["name"] == f"{task_name}_pred_head":
            action_idx = idx
            break
    del base_task_flow[action_idx]

    # ind heads
    type = "ind"

    for slice_name in slice_func_dict.keys():
        slice_ind_module_pool = base_module_pool
        slice_ind_module_pool[
            f"{task_name}_slice_{type}_{slice_name}_head"
        ] = nn.Linear(last_linear_layer_size[0], 2)
        slice_ind_task_flow = base_task_flow + [
            {
                "name": f"{task_name}_slice_{type}_{slice_name}_head",
                "module": f"{task_name}_slice_{type}_{slice_name}_head",
                "inputs": [(f"{task_name}_feature", 0)],
            }
        ]
        task = EmmentalTask(
            name=f"{task_name}_slice_{type}_{slice_name}",
            module_pool=slice_ind_module_pool,
            task_flow=slice_ind_task_flow,
            loss_func=partial(utils.ce_loss, f"{task_name}_slice_{type}_{slice_name}_head"),
            output_func=partial(utils.output, f"{task_name}_slice_{type}_{slice_name}_head"),
            scorer=Scorer(metrics=["accuracy"]),
        )
        tasks.append(task)

    # pred heads
    type = "pred"

    shared_linear_module = nn.Linear(hidden_dim, last_linear_layer_size[1])

    for slice_name in slice_func_dict.keys():
        slice_pred_module_pool = base_module_pool
        slice_pred_module_pool[f"{task_name}_slice_feat_{slice_name}"] = nn.Linear(
            last_linear_layer_size[0], hidden_dim
        )
        slice_pred_module_pool[
            f"{task_name}_slice_{type}_linear_head"
        ] = shared_linear_module
        slice_pred_task_flow = base_task_flow + [
            {
                "name": f"{task_name}_slice_feat_{slice_name}",
                "module": f"{task_name}_slice_feat_{slice_name}",
                "inputs": [(f"{task_name}_feature", 0)],
            },
            {
                "name": f"{task_name}_slice_{type}_{slice_name}_head",
                "module": f"{task_name}_slice_{type}_linear_head",
                "inputs": [(f"{task_name}_slice_feat_{slice_name}", 0)],
            },
        ]
        task = EmmentalTask(
            name=f"{task_name}_slice_{type}_{slice_name}",
            module_pool=slice_pred_module_pool,
            task_flow=slice_pred_task_flow,
            loss_func=partial(utils.ce_loss, f"{task_name}_slice_{type}_{slice_name}_head"),
            output_func=partial(utils.output, f"{task_name}_slice_{type}_{slice_name}_head"),
            scorer=base_scorer,
        )
        tasks.append(task)

    # master
    master_task = EmmentalTask(
        name=f"{task_name}",
        module_pool=nn.ModuleDict(
            {
                f"{task_name}_pred_feat": master_module.SliceMasterModule(),
                f"{task_name}_pred_head": nn.Linear(
                    hidden_dim, last_linear_layer_size[1]
                ),
            }
        ),
        task_flow=[
            {
                "name": f"{task_name}_pred_feat",
                "module": f"{task_name}_pred_feat",
                "inputs": [],
            },
            {
                "name": f"{task_name}_pred_head",
                "module": f"{task_name}_pred_head",
                "inputs": [(f"{task_name}_pred_feat", 0)],
            },
        ],
        loss_func=partial(utils.ce_loss, f"{task_name}_pred_head"),
        output_func=partial(utils.output, f"{task_name}_pred_head"),
        scorer=base_scorer,
    )
    tasks.append(master_task)

    return tasks

