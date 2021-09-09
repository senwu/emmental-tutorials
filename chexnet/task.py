from functools import partial

import torch.nn.functional as F
from modules.torch_vision_encoder import TorchVisionEncoder
from torch import nn

from emmental.scorer import Scorer
from emmental.task import EmmentalTask


def ce_loss(module_name, immediate_ouput_dict, Y, active):
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0][active], Y.view(-1)[active]
    )


def output(module_name, immediate_ouput_dict):
    return F.softmax(immediate_ouput_dict[module_name][0], dim=1)


def create_task(task_names, cnn_encoder="densenet121"):
    input_shape = (3, 224, 224)
    cnn_module = TorchVisionEncoder(cnn_encoder, pretrained=True)
    classification_layer_dim = cnn_module.get_frm_output_size(input_shape)

    tasks = []

    for task_name in task_names:
        loss_fn = partial(ce_loss, f"{task_name}_pred_head")
        output_func = partial(output, f"{task_name}_pred_head")
        scorer = Scorer(metrics=["roc_auc"])

        task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    f"feature": cnn_module,
                    f"{task_name}_pred_head": nn.Linear(classification_layer_dim, 2),
                }
            ),
            task_flow=[
                {
                    "name": f"feature",
                    "module": "feature",
                    "inputs": [("_input_", "image")],
                },
                {
                    "name": f"{task_name}_pred_head",
                    "module": f"{task_name}_pred_head",
                    "inputs": [(f"feature", 0)],
                },
            ],
            loss_func=loss_fn,
            output_func=output_func,
            scorer=scorer,
        )
        tasks.append(task)

    return tasks
