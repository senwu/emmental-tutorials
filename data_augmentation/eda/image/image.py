import logging
import sys

import torch.backends.cudnn as cudnn
from emmental import EmmentalLearner, EmmentalModel, Meta, init
from emmental.utils.parse_args import parse_args_to_config

from eda.image.augment_policy import Augmentation
from eda.image.data import get_dataloaders
from eda.image.scheduler import AugScheduler
from eda.image.task import create_task
from eda.utils import write_to_file, write_to_json_file

logger = logging.getLogger(__name__)


def main(args):
    # Initialize Emmental
    config = parse_args_to_config(args)
    init(log_dir=config["meta_config"]["log_path"], config=config)

    # Log configuration into files
    cmd_msg = " ".join(sys.argv)
    logger.info(f"COMMAND: {cmd_msg}")
    write_to_file(f"{Meta.log_path}/cmd.txt", cmd_msg)

    logger.info(f"Config: {Meta.config}")
    write_to_file(f"{Meta.log_path}/config.txt", Meta.config)

    # Create dataloaders
    dataloaders = get_dataloaders(args)

    # Assign transforms to dataloaders
    aug_dataloaders = []
    if args.augment_policy:
        for idx in range(len(dataloaders)):
            if dataloaders[idx].split in args.train_split:
                dataloaders[idx].dataset.transform_cls = Augmentation(args=args)

    config["learner_config"]["task_scheduler_config"]["task_scheduler"] = AugScheduler(
        augment_k=args.augment_k, enlarge=args.augment_enlarge
    )
    Meta.config["learner_config"]["task_scheduler_config"]["task_scheduler"] = config[
        "learner_config"
    ]["task_scheduler_config"]["task_scheduler"]

    # Create tasks
    model = EmmentalModel(name=f"{args.task}_task")
    model.add_task(create_task(args))

    # Set cudnn benchmark
    cudnn.benchmark = True

    # Load the best model from the pretrained model
    if config["model_config"]["model_path"] is not None:
        model.load(config["model_config"]["model_path"])

    if args.train:
        emmental_learner = EmmentalLearner()
        emmental_learner.learn(model, dataloaders + aug_dataloaders)

    # Remove all extra augmentation policy
    for idx in range(len(dataloaders)):
        dataloaders[idx].dataset.transform_cls = None

    scores = model.score(dataloaders)

    # Save metrics and models
    logger.info(f"Metrics: {scores}")
    scores["log_path"] = Meta.log_path
    write_to_json_file(f"{Meta.log_path}/metrics.txt", scores)
    model.save(f"{Meta.log_path}/last_model.pth")
