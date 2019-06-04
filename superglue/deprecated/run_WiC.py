import logging
from functools import partial

import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from modules.bert_module import BertModule
from parse_WiC_slice import get_WiC_dataloaders
from slice_WiC import slice_func_dict
from task_config import SuperGLUE_LABEL_MAPPING, SuperGLUE_TASK_METRIC_MAPPING
from sklearn.metrics import f1_score
import os

TASK_NAME = "WiC"
DATA_DIR = os.environ["SUPERGLUEDATA"]
BERT_MODEL_NAME = "bert-large-cased"
BATCH_SIZE = 4

BERT_OUTPUT_DIM = 768 if "base" in BERT_MODEL_NAME else 1024
TASK_CARDINALITY = (
    len(SuperGLUE_LABEL_MAPPING[TASK_NAME].keys())
    if SuperGLUE_LABEL_MAPPING[TASK_NAME] is not None
    else 1
)


logger = logging.getLogger(__name__)

emmental.init(
    "logs",
    config={
        "model_config": {"device": 0, "dataparallel": False},
        "learner_config": {
            "n_epochs": 1,
            "valid_split": "val",
            "optimizer_config": {"optimizer": "adam", "lr": 1e-5},
            "min_lr": 0,
            "lr_scheduler_config": {"warmup_percentage": 0.1, "lr_scheduler": None},
        },
        "logging_config": {
            "counter_unit": "epoch",
            "evaluation_freq": 0.1,
            "checkpointing": True,
            "checkpointer_config": {"checkpoint_metric": {"WiC/SuperGLUE/val/accuracy":"max"}},
        },
    },
)

dataloaders = get_WiC_dataloaders(
    data_dir=DATA_DIR,
    task_name=TASK_NAME,
    splits=["train", "val", "test"],
    max_sequence_length=128,
    max_data_samples=None,
    tokenizer_name=BERT_MODEL_NAME,
    batch_size=BATCH_SIZE,
    slice_func_dict=slice_func_dict,
)

print(dataloaders["train"].dataset.Y_dict)
print(dataloaders["train"].task_to_label_dict)
for key, value in dataloaders["train"].dataset.Y_dict.items():
    print(key, value.size())

# Build Emmental task
def ce_loss(module_name, immediate_ouput_dict, Y, active):
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0][active], (Y.view(-1) - 1)[active]
    )

def output(module_name, immediate_ouput_dict):
    return F.softmax(immediate_ouput_dict[module_name][0], dim=1)

class FeatureConcateModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature, idx1, idx2):
        last_layer = feature[-1]
        emb = last_layer[:,0,:]
        idx1 = idx1.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, last_layer.size(-1)])
        idx2 = idx2.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, last_layer.size(-1)])
        word1_emb = last_layer.gather(dim=1, index=idx1).squeeze(dim=1)
        word2_emb = last_layer.gather(dim=1, index=idx2).squeeze(dim=1)
        input = torch.cat([emb, word1_emb, word2_emb], dim=-1)
        return input

class SliceModule(nn.Module):
    def __init__(self, feature_dim, class_cardinality):
        super().__init__()
        self.linear = nn.Linear(feature_dim, class_cardinality)

    def forward(self, feature):
        return self.linear.forward(feature)

class MasterModule(nn.Module):
        def __init__(self, feature_dim, class_cardinality):
            super().__init__()
            self.linear = nn.Linear(feature_dim, class_cardinality)

        def forward(self, immediate_ouput_dict):
            slice_ind_names = sorted(
                [
                    flow_name
                    for flow_name in immediate_ouput_dict.keys()
                    if "_slice_ind_" in flow_name
                ]
            )
            slice_pred_names = sorted(
                [
                    flow_name
                    for flow_name in immediate_ouput_dict.keys()
                    if "_slice_pred_" in flow_name
                ]
            )

            Q = torch.cat(
                [
                    F.softmax(immediate_ouput_dict[slice_ind_name][0])[:, 0].unsqueeze(1)
                    for slice_ind_name in slice_ind_names
                ],
                dim=-1,
            )
            P = torch.cat(
                [
                    F.softmax(immediate_ouput_dict[slice_pred_name][0])[:, 0].unsqueeze(1)
                    for slice_pred_name in slice_pred_names
                ],
                dim=-1,
            )

            slice_feat_names = sorted(
                [
                    flow_name
                    for flow_name in immediate_ouput_dict.keys()
                    if "_slice_feat_" in flow_name
                ]
            )

            slice_reps = torch.cat(
                [
                    immediate_ouput_dict[slice_feat_name][0].unsqueeze(1)
                    for slice_feat_name in slice_feat_names
                ],
                dim=1,
            )

            A = F.softmax(Q * P, dim=1).unsqueeze(-1).expand([-1, -1, slice_reps.size(-1)])

            reweighted_rep = torch.sum(A * slice_reps, 1)

            return self.linear.forward(reweighted_rep)

def build_tasks(slice_func_dict):
    H = BERT_OUTPUT_DIM
    shared_classification_module = nn.Linear(H, TASK_CARDINALITY)
    bert_module = BertModule(BERT_MODEL_NAME)

    tasks = []

    # Add ind task
    type = "ind"

    for slice_name in slice_func_dict.keys():
        task = EmmentalTask(
            name=f"{TASK_NAME}_slice_{type}_{slice_name}",
            module_pool=nn.ModuleDict(
                {
                    "feature": FeatureConcateModule(),
                    f"{TASK_NAME}_slice_{type}_{slice_name}_head": SliceModule(
                        3 * BERT_OUTPUT_DIM, 2
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
                    "name": f"feature",
                    "module": f"feature",
                    "inputs": [
                        ("input", 0),
                        ("_input_", "sent1_idxs"),
                        ("_input_", "sent2_idxs"),
                    ],
                },
                {
                    "name": f"{TASK_NAME}_slice_{type}_{slice_name}_head",
                    "module": f"{TASK_NAME}_slice_{type}_{slice_name}_head",
                    "inputs": [("feature", 0)],
                },
            ],
            loss_func=partial(ce_loss, f"{TASK_NAME}_slice_{type}_{slice_name}_head"),
            output_func=partial(output, f"{TASK_NAME}_slice_{type}_{slice_name}_head"),
            scorer=Scorer(metrics=["accuracy"]),
        )
        tasks.append(task)

    # Add pred task
    type = "pred"
    for slice_name in slice_func_dict.keys():
        task = EmmentalTask(
            name=f"{TASK_NAME}_slice_{type}_{slice_name}",
            module_pool=nn.ModuleDict(
                {
    #                 "bert_module": bert_module,
                    "feature": FeatureConcateModule(),
                    f"{TASK_NAME}_slice_feat_{slice_name}": nn.Linear(3 * BERT_OUTPUT_DIM, H),
                    f"{TASK_NAME}_slice_{type}_{slice_name}_head": shared_classification_module,
                }
            ),
            task_flow=[
                {
                    "name": "input",
                    "module": "bert_module",
                    "inputs": [("_input_", "token_ids"), ("_input_", "token_segments")],
                },
                {
                    "name": f"feature",
                    "module": f"feature",
                    "inputs": [
                        ("input", 0),
                        ("_input_", "sent1_idxs"),
                        ("_input_", "sent2_idxs"),
                    ],
                },
                {
                    "name": f"{TASK_NAME}_slice_feat_{slice_name}",
                    "module": f"{TASK_NAME}_slice_feat_{slice_name}",
                    "inputs": [("feature", 0)],
                },
                {
                    "name": f"{TASK_NAME}_slice_{type}_{slice_name}_head",
                    "module": f"{TASK_NAME}_slice_{type}_{slice_name}_head",
                    "inputs": [(f"{TASK_NAME}_slice_feat_{slice_name}", 0)],
                },
            ],
            loss_func=partial(ce_loss, f"{TASK_NAME}_slice_{type}_{slice_name}_head"),
            output_func=partial(output, f"{TASK_NAME}_slice_{type}_{slice_name}_head"),
            scorer=Scorer(metrics=SuperGLUE_TASK_METRIC_MAPPING[TASK_NAME]),
        )
        tasks.append(task)

    master_task = EmmentalTask(
        name=f"{TASK_NAME}",
        module_pool=nn.ModuleDict(
            {
                "bert_module": bert_module,
                f"{TASK_NAME}_pred_head": MasterModule(H, TASK_CARDINALITY),
            }
        ),
        task_flow=[
            {
                "name": f"{TASK_NAME}_pred_head",
                "module": f"{TASK_NAME}_pred_head",
                "inputs": [],
            }
        ],
        loss_func=partial(ce_loss, f"{TASK_NAME}_pred_head"),
        output_func=partial(output, f"{TASK_NAME}_pred_head"),
        scorer=Scorer(metrics=SuperGLUE_TASK_METRIC_MAPPING[TASK_NAME]),
    )
    tasks.append(master_task)
    return tasks

tasks = build_tasks(slice_func_dict)
mtl_model = EmmentalModel(name="SuperGLUE_single_task", tasks=tasks)
emmental_learner = EmmentalLearner()
emmental_learner.learn(mtl_model, dataloaders.values())
logging.info(mtl_model.score(dataloaders["val"]))