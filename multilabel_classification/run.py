# Copyright (c) 2021 Sen Wu. All Rights Reserved.


import argparse
import logging
import sys

import emmental
from dataset import ToxicCommentDataset
from emmental import Meta
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_args import parse_args, parse_args_to_config
from emmental.utils.utils import str2bool, str2list
from task import create_task
from utils import write_to_file, write_to_json_file
from transformers import AutoTokenizer
import os 
logger = logging.getLogger(__name__)


def add_application_args(parser):

    # Application configuration
    application_config = parser.add_argument_group("Application configuration")

    parser.add_argument("--data_path", type=str, help="The path to data files")

    parser.add_argument(
        "--model", type=str, default="distilbert-base-uncased", help="The model to use"
    )

    parser.add_argument("--batch_size", type=int, default=5, help="batch size")

    parser.add_argument(
        "--max_data_samples", type=int, default=0, help="Maximum data samples to use"
    )

    parser.add_argument(
        "--train", type=str2bool, default=True, help="Whether to train the model"
    )


if __name__ == "__main__":
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        "Toxic Comment Runner", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser = parse_args(parser=parser)
    add_application_args(parser)

    args = parser.parse_args()
    config = parse_args_to_config(args)
    emmental.init(config["meta_config"]["log_path"], config=config)

    cmd_msg = " ".join(sys.argv)
    logger.info(f"COMMAND: {cmd_msg}")
    write_to_file(f"{Meta.log_path}/cmd.txt", cmd_msg)
    logger.info(f"Config: {Meta.config}")
    write_to_file(f"{Meta.log_path}/config.txt", Meta.config)

    # task_to_label_dict = {task_name: task_name for task_name in args.task}

    dataloaders = []
    tasks = []

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    # train_split_ids = 

    for split in ["train", "test"]:
        dataset = ToxicCommentDataset(
            name="toxic",
            file_path=os.path.join(args.data_path, f"train.csv.zip"),
            id_file_path=os.path.join(args.data_path, f"{split}_ids.csv"),
            split=split,
            tokenizer=tokenizer,
            max_data_samples=args.max_data_samples if split == "train" else None,
            max_length=128,
        )
        logger.info(f"Loaded {split} containing {len(dataset)} samples.")
        # logger.info(dataset[0])
        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict={"toxic": "labels"},
                dataset=dataset,
                split=split,
                shuffle=True if split == "train" else False,
                batch_size=args.batch_size,
                # num_workers=1,
            )
        )
        logger.info(f"Built dataloader for {dataset.name} {split} set.")

    tasks = create_task(args)

    # Build Emmental model
    model = EmmentalModel(name="toxic", tasks=tasks)

    # Load the pre-trained model
    if Meta.config["model_config"]["model_path"]:
        model.load(Meta.config["model_config"]["model_path"])

    # Training
    if args.train:
        emmental_learner = EmmentalLearner()
        emmental_learner.learn(model, dataloaders)

    scores = model.score(dataloaders)

    # Save metrics into file
    logger.info(f"Metrics: {scores}")
    write_to_json_file(f"{Meta.log_path}/metrics.txt", scores)

    # Save best metrics into file
    if args.checkpointing:
        logger.info(
            f"Best metrics: "
            f"{emmental_learner.logging_manager.checkpointer.best_metric_dict}"
        )
        write_to_file(
            f"{Meta.log_path}/best_metrics.txt",
            emmental_learner.logging_manager.checkpointer.best_metric_dict,
        )
