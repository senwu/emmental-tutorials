from functools import partial
import json
import os

import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch import nn
import torch.nn.functional as F

import emmental
from emmental.model import EmmentalModel
from emmental import Meta
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from modules.bert_module import BertModule
from task_config import (SuperGLUE_LABEL_MAPPING, SuperGLUE_TASK_METRIC_MAPPING, SuperGLUE_TASK_SPLIT_MAPPING)
from parse_WiC import get_WiC_dataloaders

PKL_PATH = "/dfs/scratch0/bradenjh/emmental-tutorials/superglue/models/WiC_best_v0.pth"
SPLIT = "val"

TASK_NAME = "WiC"
DATA_DIR = os.environ["SUPERGLUEDATA"]
TUTORIALS_ROOT = "/dfs/scratch0/bradenjh/emmental-tutorials/"
BERT_MODEL_NAME = "bert-large-cased"
BATCH_SIZE = 4

emmental.init(
    "logs",
    config={
        "model_config": {"device": 0, "dataparallel": False},
        "learner_config": {
            "n_epochs": 4,
            "valid_split": "val",
            "optimizer_config": {"optimizer": "adam", "lr": 1e-5},
            "min_lr": 0,
            "lr_scheduler_config": {
                "warmup_percentage": 0.1,
                "lr_scheduler": None,
            },
        },
        "logging_config": {
            "counter_unit": "batch",
            "evaluation_freq": 10,
            "checkpointing": True,
            "checkpointer_config": {
                "checkpoint_metric": {"WiC/SuperGLUE/val/accuracy":"max"},
                "checkpoint_freq": 1,
            },
        },
    },
)

dataloaders = get_WiC_dataloaders(
    data_dir=DATA_DIR,
    task_name=TASK_NAME,
    splits=["train", "val", "test"],
    max_sequence_length=256,
    max_data_samples=None,
    tokenizer_name=BERT_MODEL_NAME,
    batch_size=BATCH_SIZE,
)

class LinearModule(nn.Module):
    def __init__(self, feature_dim, class_cardinality):
        super().__init__()

        self.linear = nn.Linear(feature_dim, class_cardinality)

    def forward(self, feature, idx1, idx2):
        last_layer = feature[-1]
        emb = last_layer[:,0,:]
        idx1 = idx1.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, last_layer.size(-1)])
        idx2 = idx2.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, last_layer.size(-1)])
        word1_emb = last_layer.gather(dim=1, index=idx1).squeeze(dim=1)
        word2_emb = last_layer.gather(dim=1, index=idx2).squeeze(dim=1)
        input = torch.cat([emb, word1_emb, word2_emb], dim=-1)
        return self.linear.forward(input)

def ce_loss(task_name, immediate_ouput_dict, Y, active):
    module_name = f"{task_name}_pred_head"
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0][active], (Y.view(-1) - 1)[active]
    )

def output(task_name, immediate_ouput_dict):
    module_name = f"{task_name}_pred_head"
    return F.softmax(immediate_ouput_dict[module_name][0], dim=1)

def macro_f1(golds, probs, preds):
    return {"macro_f1": f1_score(golds, preds, average="macro")}

def make_analysis_df(model):
    # Get predictions
    gold_dict, prob_dict, pred_dict = model.predict(dataloaders[SPLIT], return_preds=True)
    probs = prob_dict["WiC"][:,0]
    preds = pred_dict["WiC"]

    # Load raw data
    jsonl_path = os.path.join(
        DATA_DIR, TASK_NAME, SuperGLUE_TASK_SPLIT_MAPPING[TASK_NAME][SPLIT]
    )

    # Add new columns
    rows = [json.loads(row) for row in open(jsonl_path, encoding="utf-8")]
    for i, row in enumerate(rows):
        row["prob"] = probs[i]
        row["pred"] = True if preds[i] == 1 else False
        row["correct"] = "Y" if row["pred"] == row["label"] else "N"

    # Make tsv
    df = pd.DataFrame(rows)
    df = df[['idx', 'label', 'pred', 'prob', 'correct', 'word', 'pos', 'sentence1', 'sentence2']]
    return df

# Load model and sanity check quality
# model = get_and_load_model()
model = EmmentalModel()
model.load(PKL_PATH)
print(model.score(dataloaders["val"]))

# Make df and write to file
df = make_analysis_df(model)
out_path = os.path.join(TUTORIALS_ROOT, "superglue", "analysis", "WiC_analysis_v0.csv")
df.to_csv(out_path)
print(f"Wrote error analysis to {out_path}")


