import argparse
import logging
import sys

from dataset import CXR8Dataset
from task import create_task
from transforms import get_data_transforms
from utils import write_to_file, write_to_json_file

import emmental
from emmental import Meta
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_args import parse_args, parse_args_to_config
from emmental.utils.utils import str2bool, str2list

logger = logging.getLogger(__name__)


def add_application_args(parser):

    # Application configuration
    application_config = parser.add_argument_group("Application configuration")

    parser.add_argument("--data_path", type=str, help="The path to csv file")

    parser.add_argument("--image_path", type=str, help="The path to image files")

    parser.add_argument(
        "--model", type=str, default="densenet121", help="The model to use"
    )

    application_config.add_argument(
        "--task", type=str2list, required=True, help="Image classification tasks"
    )

    parser.add_argument("--batch_size", type=int, default=5, help="batch size")

    parser.add_argument(
        "--max_data_samples", type=int, default=0, help="Maximum data samples to use"
    )

    parser.add_argument(
        "--slices", type=str2bool, default=False, help="Whether to include slices"
    )

    parser.add_argument(
        "--train", type=str2bool, default=True, help="Whether to train the model"
    )


if __name__ == "__main__":
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        "ChexNet Runner", formatter_class=argparse.ArgumentDefaultsHelpFormatter
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

    DATA_NAME = "CXR8"

    task_to_label_dict = {task_name: task_name for task_name in args.task}

    cxr8_transform = get_data_transforms(DATA_NAME)

    dataloaders = []
    tasks = []

    for split in ["train", "val", "test"]:
        dataset = CXR8Dataset(
            name=DATA_NAME,
            path_to_images=args.image_path,
            path_to_labels=args.data_path,
            split=split,
            transform=cxr8_transform[split],
            sample=args.max_data_samples,
        )
        logger.info(
            f"Loaded {split} for {DATA_NAME} containing {len(dataset)} samples."
        )
        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict=task_to_label_dict,
                dataset=dataset,
                split=split,
                shuffle=True if split == "train" else False,
                batch_size=args.batch_size,
                num_workers=8,
            )
        )
        logger.info(f"Built dataloader for {dataset.name} {split} set.")

    tasks = create_task(list(task_to_label_dict.keys()), cnn_encoder=args.model)

    # Build Emmental model
    model = EmmentalModel(name=DATA_NAME, tasks=tasks)

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
