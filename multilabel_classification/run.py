# Copyright (c) 2021 Sen Wu. All Rights Reserved.


import argparse
import logging
import sys

import emmental
from emmental import Meta
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_args import parse_args, parse_args_to_config
from emmental.utils.utils import nullable_string, str2bool, str2list,nullable_int
from transformers import AutoTokenizer

from dataset import MultiLabelDataset
from task import create_task
from utils import write_to_file, write_to_json_file

logger = logging.getLogger(__name__)


def add_application_args(parser):

    # Application configuration
    application_config = parser.add_argument_group("Application configuration")

    application_config.add_argument("--task_name", type=str, help="Task name")

    application_config.add_argument(
        "--train_data_path", type=nullable_string, help="The path to train csv file"
    )

    application_config.add_argument(
        "--val_data_path", type=nullable_string, help="The path to val csv file"
    )

    application_config.add_argument(
        "--test_data_path", type=nullable_string, help="The path to test csv file"
    )

    application_config.add_argument(
        "--input_field", type=str, help="The input field name"
    )

    application_config.add_argument(
        "--label_fields", type=str2list, help="The label field names"
    )

    application_config.add_argument(
        "--model", type=str, default="distilbert-base-uncased", help="The model to use"
    )

    application_config.add_argument(
        "--batch_size", type=int, default=5, help="batch size"
    )

    application_config.add_argument(
        "--max_data_samples", type=nullable_int, default=None, help="Maximum data samples to use"
    )

    application_config.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum sequence length of model",
    )

    application_config.add_argument(
        "--train", type=str2bool, default=True, help="Whether to train the model"
    )


if __name__ == "__main__":
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        "Application Runner", formatter_class=argparse.ArgumentDefaultsHelpFormatter
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

    dataloaders = []

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    for split, data_path in [
        ("train", args.train_data_path),
        ("val", args.val_data_path),
        ("test", args.test_data_path),
    ]:
        if data_path is None:
            continue
        dataset = MultiLabelDataset(
            name=args.task_name,
            data_path=data_path,
            input_field=args.input_field,
            label_fields=args.label_fields,
            split=split,
            tokenizer=tokenizer,
            max_data_samples=args.max_data_samples, # if split == "train" else None,
            max_seq_length=args.max_seq_length,
        )
        logger.info(f"Loaded {split} containing {len(dataset)} samples.")
        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict={args.task_name: "labels"},
                dataset=dataset,
                split=split,
                shuffle=True if split == "train" else False,
                batch_size=args.batch_size,
                # num_workers=8,
            )
        )
        logger.info(f"Built dataloader for {dataset.name} {split} set.")

    # Build Emmental model
    model = EmmentalModel(name=args.task_name, tasks=create_task(args))

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
