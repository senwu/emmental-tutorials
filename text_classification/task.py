from functools import partial

import torch.nn.functional as F
from modules import CNN, LSTM, Average
from torch import nn

from emmental.modules.identity_module import IdentityModule
from emmental.scorer import Scorer
from emmental.task import EmmentalTask


def ce_loss(module_name, immediate_ouput_dict, Y, active):
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0][active], Y.view(-1)[active]
    )


def output(module_name, immediate_ouput_dict):
    return F.softmax(immediate_ouput_dict[module_name][0])


def create_task(task_name, args, nclasses, emb_layer):
    if args.model == "cnn":
        input_module = IdentityModule()
        feature_extractor = CNN(emb_layer.n_d, widths=[3, 4, 5], filters=args.n_filters)
        d_out = args.n_filters * 3
    elif args.model == "lstm":
        input_module = IdentityModule()
        feature_extractor = LSTM(
            emb_layer.n_d, args.dim, args.depth, dropout=args.dropout
        )
        d_out = args.dim
    elif args.model == "mlp":
        input_module = Average()
        feature_extractor = nn.Sequential(nn.Linear(emb_layer.n_d, args.dim), nn.ReLU())
        d_out = args.dim
    else:
        raise ValueError(f"Unrecognized model {args.model}.")

    return EmmentalTask(
        name=task_name,
        module_pool=nn.ModuleDict(
            {
                "emb": emb_layer,
                "input": input_module,
                "feature": feature_extractor,
                "dropout": nn.Dropout(args.dropout),
                f"{task_name}_pred_head": nn.Linear(d_out, nclasses),
            }
        ),
        task_flow=[
            {"name": "emb", "module": "emb", "inputs": [("_input_", "feature")]},
            {
                "name": "input",
                "module": "input",
                "inputs": [("emb", 0)],
            },
            {"name": "feature", "module": "feature", "inputs": [("input", 0)]},
            {"name": "dropout", "module": "dropout", "inputs": [("feature", 0)]},
            {
                "name": f"{task_name}_pred_head",
                "module": f"{task_name}_pred_head",
                "inputs": [("dropout", 0)],
            },
        ],
        loss_func=partial(ce_loss, f"{task_name}_pred_head"),
        output_func=partial(output, f"{task_name}_pred_head"),
        scorer=Scorer(metrics=["accuracy"]),
    )
