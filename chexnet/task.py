from functools import partial

import torch.nn.functional as F
from modules.torch_vision_encoder import TorchVisionEncoder
from torch import nn

from emmental import Action, EmmentalTask, Scorer


def ce_loss(module_name, output_dict, Y):
    return F.cross_entropy(output_dict[module_name], Y)


def output(module_name, output_dict):
    return F.softmax(output_dict[module_name], dim=1)


def create_task(task_names, cnn_encoder="densenet121"):
    input_shape = (3, 224, 224)
    cnn_module = TorchVisionEncoder(cnn_encoder, pretrained=True)
    classification_layer_dim = cnn_module.get_frm_output_size(input_shape)

    tasks = [
        EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "feature": cnn_module,
                    f"{task_name}_pred_head": nn.Linear(classification_layer_dim, 2),
                }
            ),
            task_flow=[
                Action(
                    name="feature",
                    module="feature",
                    inputs=[("_input_", "image")],
                ),
                Action(
                    name=f"{task_name}_pred_head",
                    module=f"{task_name}_pred_head",
                    inputs=[("feature", 0)],
                ),
            ],
            loss_func=partial(ce_loss, f"{task_name}_pred_head"),
            output_func=partial(output, f"{task_name}_pred_head"),
            scorer=Scorer(metrics=["roc_auc"]),
        )
        for task_name in task_names
    ]

    return tasks
