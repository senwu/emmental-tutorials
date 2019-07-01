load_ext autoreload
autoreload 2

import logging

import argparse
from functools import partial
import torch.nn as nn
import torch.nn.functional as F

import emmental
from cxr_dataset import CXR8Dataset
from emmental import Meta
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from modules.classification_module import ClassificationModule
from modules.torch_vision_encoder import TorchVisionEncoder
from task_config import CXR8_TASK_NAMES
from transforms import get_data_transforms

# Initializing logs
logger = logging.getLogger(__name__)
emmental.init("logs")

# Defining helper functions
def ce_loss(task_name, immediate_ouput, Y, active):
    return F.cross_entropy(
        immediate_ouput[f"classification_module_{task_name}"][0], Y.view(-1) - 1
    )

def output(task_name, immediate_ouput):
    return F.softmax(immediate_ouput[f"classification_module_{task_name}"][0], dim=1)

def parse_args():
    parser = argparse.ArgumentParser(description='Run chexnet slicing experiments')
    parser.add_argument('--data_name', default='CXR8', help='Dataset name')
    parser.add_argument('cxrdata_path', 
                        default=f"/dfs/scratch1/senwu/mmtl/emmental-tutorials/chexnet/data/nih_labels.csv",
                        help='Path to labels')
    parser.add_argument('--cxrimage_path', 
                        default=f"/dfs/scratch1/senwu/mmtl/emmental-tutorials/chexnet/data/images",
                        help='Path to images')
    parser.add_argument('--tasks', default='CXR8', type=str, nargs='+',
                        help='list of tasks; if CXR8, all CXR8; if TRIAGE, normal/abnormal')

    args = parser.parse_args()
    return args

if __name__=="__main__":
    # Parsing command line arguments
    args = parse_args()
        
    # Configuring run data
    Meta.update_config(
        config={
            "meta_config": {"seed": 1701, "device": 0},
            "learner_config": {
                "n_epochs": 20,
                "valid_split": "val",
                "optimizer_config": {"optimizer": "sgd", "lr": 0.001, "l2": 0.000},
                "lr_scheduler_config": {
                    "warmup_steps": None,
                    "warmup_unit": "batch",
                    "lr_scheduler": "linear",
                    "min_lr": 1e-6,
                },
            },
            "logging_config": {"evaluation_freq": 4000, "checkpointing": False},
        }
    )

    # Getting paths to data
    DATA_NAME = args.data_name
    CXRDATA_PATH = args.cxrdata_path
    CXRIMAGE_PATH = args.cxrimage_path

    # Providing model settings
    BATCH_SIZE = 16
    CNN_ENCODER = "densenet121"

    BATCH_SIZES = {"train": 16, "val": 64, "test": 64}

    # Getting transforms
    cxr8_transform = get_data_transforms(DATA_NAME)

    # Creating datasets
    datasets = {}

    for split in ["train", "val", "test"]:

        datasets[split] = CXR8Dataset(
            name=DATA_NAME,
            path_to_images=CXRIMAGE_PATH,
            path_to_labels=CXRDATA_PATH,
            split=split,
            transform=cxr8_transform[split],
            sample=0,
            seed=1701,
        )

        logger.info(f"Loaded {split} split for {DATA_NAME}.")

    # Getting task to label dict
    # All possible tasks in dataloader
    all_tasks = CXR8_TASK_NAMES +['Abnormal']
    if args.tasks == 'CXR8':
        # Standard chexnet
        task_list = CXR8_TASK_NAMES:
    elif:
        # Binary triage
        args.tasks == 'TRIAGE':
        task_list = ['Abnormal']
    else:
        # Otherwise, making sure tasks are valid
        task_list = args.tasks
        for task in task_list:
            assert(task in all_tasks)

    task_to_label_dict = {task_name: task_name for task_name in task_list}
    print(task_to_label_dict)

    # Building dataloaders
    dataloaders = []

    for split in ["train", "val", "test"]:
        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict=task_to_label_dict,
                dataset=datasets[split],
                split=split,
                shuffle=True if split == "train" else False,
                batch_size=BATCH_SIZES[split],
                num_workers=8,
            )
        )
        logger.info(f"Built dataloader for {datasets[split].name} {split} set.")


    # Building Emmental tasks
    input_shape = (3, 224, 224)

    cnn_module = TorchVisionEncoder(CNN_ENCODER, pretrained=True)
    classification_layer_dim = cnn_module.get_frm_output_size(input_shape)

    tasks = [
        EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "cnn": cnn_module,
                    f"classification_module_{task_name}": ClassificationModule(
                        classification_layer_dim, 2
                    ),
                }
            ),
            task_flow=[
                {"name": "cnn", "module": "cnn", "inputs": [("_input_", "image")]},
                {
                    "name": f"classification_module_{task_name}",
                    "module": f"classification_module_{task_name}",
                    "inputs": [("cnn", 0)],
                },
            ],
            loss_func=partial(ce_loss, task_name),
            output_func=partial(output, task_name),
            scorer=Scorer(metrics=["accuracy", "roc_auc"]),
        )
        for task_name in task_list
    ]

    # Defining model and trainer
    mtl_model = EmmentalModel(name="Chexnet", tasks=tasks)
    emmental_learner = EmmentalLearner()

    # Training model
    emmental_learner.learn(mtl_model, dataloaders)