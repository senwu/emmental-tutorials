import logging

from . import (
    # CB_slices,
    # COPA_slices,
    # MultiRC_slices,
    # RTE_slices,
    WiC_slices,
    # WSC_slices,
)

slice_func_dict = {
    # "CB": CB_slices.slice_func_dict,
    # "COPA": COPA_slices.slice_func_dict,
    # "MultiRC": MultiRC_slices.slice_func_dict,
    # "RTE": RTE_slices.slice_func_dict,
    "WiC": WiC_slices.slice_func_dict,
    # "WSC": WSC_slices.slice_func_dict,
}

logger = logging.getLogger(__name__)

def add_slice_labels(task_name, dataloaders, slice_func_dict):
    for slice_name, slice_func in slice_func_dict.items():
        for dataloader in dataloaders:
            ind, pred = slice_func(dataloader.dataset)
            dataloader.dataset.Y_dict.update({f"{task_name}_slice_ind_{slice_name}": ind, f"{task_name}_slice_pred_{slice_name}": pred})
            dataloader.task_to_label_dict.update({f"{task_name}_slice_ind_{slice_name}": f"{task_name}_slice_ind_{slice_name}", f"{task_name}_slice_pred_{slice_name}": f"{task_name}_slice_pred_{slice_name}"})    
        msg = (f"Loaded slice labels for task {task_name}, slice {slice_name}, "
               f"split {dataloader.split}.")
        logger.info(msg)

    return dataloaders


def add_slice_tasks(task_name, base_task, slice_func_dict):
    # TODO(senwu)
    raise NotImplementedError