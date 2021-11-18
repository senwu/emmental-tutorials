import logging
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from emmental import Action, EmmentalTask, Scorer
from torch import nn

from eda.image.config import TASK_INPUT_SIZE, TASK_METRIC, TASK_NUM_CLASS
from eda.image.models import ALL_MODELS
from eda.image.modules.soft_cross_entropy_loss import SoftCrossEntropyLoss

logger = logging.getLogger(__name__)


SCE = SoftCrossEntropyLoss(reduction="none")


def sce_loss(module_name, output_dict, Y):
    if len(Y.size()) == 1:
        label = output_dict[module_name].new_zeros(output_dict[module_name].size())
        label.scatter_(1, Y.view(Y.size()[0], 1), 1.0)
    else:
        label = Y

    return SCE(output_dict[module_name], label)


def output_classification(module_name, output_dict):
    return F.softmax(output_dict[module_name], dim=1)


def create_task(args):
    task_name = args.task
    n_class = TASK_NUM_CLASS[args.task]

    if args.model in ["wide_resnet"]:
        feature_extractor = ALL_MODELS[args.model](
            args.wide_resnet_depth,
            args.wide_resnet_width,
            args.wide_resnet_dropout,
            n_class,
            has_fc=False,
        )
        n_hidden_dim = feature_extractor(
            torch.randn(TASK_INPUT_SIZE[args.task])
        ).size()[-1]

    elif args.model == "mlp":
        n_hidden_dim = args.mlp_hidden_dim
        input_dim = np.prod(TASK_INPUT_SIZE[args.task])
        feature_extractor = ALL_MODELS[args.model](
            input_dim, n_hidden_dim, n_class, has_fc=False
        )
    else:
        raise ValueError(f"Invalid model {args.model}")

    loss = sce_loss
    output = output_classification

    logger.info(f"Built model: {feature_extractor}")

    return EmmentalTask(
        name=args.task,
        module_pool=nn.ModuleDict(
            {
                "feature": feature_extractor,
                f"{task_name}_pred_head": nn.Linear(n_hidden_dim, n_class),
            }
        ),
        task_flow=[
            Action(name="feature", module="feature", inputs=[("_input_", "image")]),
            Action(
                name=f"{task_name}_pred_head",
                module=f"{task_name}_pred_head",
                inputs=[("feature", 0)],
            ),
        ],
        loss_func=partial(loss, f"{task_name}_pred_head"),
        output_func=partial(output, f"{task_name}_pred_head"),
        scorer=Scorer(metrics=TASK_METRIC[task_name]),
    )
