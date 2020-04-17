import argparse
import logging
import sys

import torch

import emmental
from data import create_dataloaders, load_data
from dataloader import load_embedding
from emmental import Meta
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_args import parse_args, parse_args_to_config
from emmental.utils.utils import nullable_string, str2bool, str2list
from modules import EmbeddingLayer
from task import create_task
from utils import write_to_file, write_to_json_file

logger = logging.getLogger(__name__)


def add_application_args(parser):

    # Application configuration
    application_config = parser.add_argument_group("Application configuration")

    application_config.add_argument(
        "--task", type=str2list, required=True, help="Text classification tasks"
    )

    application_config.add_argument(
        "--data_dir", type=str, default="data", help="The path to dataset"
    )

    application_config.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "lstm", "mlp"],
        help="Which model to use",
    )

    application_config.add_argument(
        "--embedding",
        type=nullable_string,
        default=None,
        help="Which embedding file to use",
    )

    application_config.add_argument("--dim", type=int, default=300, help="Feature dim")

    application_config.add_argument(
        "--cv", type=int, default=0, help="Which fold of the cross validation"
    )

    application_config.add_argument(
        "--batch_size", type=int, default=32, help="batch size"
    )

    application_config.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout"
    )

    application_config.add_argument(
        "--n_filters", type=int, default=100, help="Number of filters"
    )

    application_config.add_argument(
        "--depth", type=int, default=2, help="Depth of LSTM"
    )

    application_config.add_argument(
        "--fix_emb", type=str2bool, default=False, help="Fix word embedding or not"
    )


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        "Text Classification Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser = parse_args(parser=parser)
    add_application_args(parser)

    args = parser.parse_args()
    config = parse_args_to_config(args)

    emmental.init(config["meta_config"]["log_path"], config=config)

    # Log configuration into files
    cmd_msg = " ".join(sys.argv)
    logger.info(f"COMMAND: {cmd_msg}")
    write_to_file(f"{Meta.log_path}/cmd.txt", cmd_msg)

    logger.info(f"Config: {Meta.config}")
    write_to_file(f"{Meta.log_path}/config.txt", Meta.config)

    datasets = {}
    data = []

    for task_name in args.task:
        dataset, task_data = load_data(args.data_dir, task_name, args.cv)
        datasets[task_name] = dataset
        data += task_data

    emb_layer = EmbeddingLayer(
        args.dim, data, embs=load_embedding(args.embedding), fix_emb=args.fix_emb
    )

    dataloaders = []
    for task_name in args.task:
        dataloaders += create_dataloaders(
            task_name, datasets[task_name], args.batch_size, emb_layer.word2id
        )

    tasks = {
        task_name: create_task(
            task_name, args, datasets[task_name]["nclasses"], emb_layer
        )
        for task_name in args.task
    }

    model = EmmentalModel(name="TC_task")

    if Meta.config["model_config"]["model_path"]:
        model.load(Meta.config["model_config"]["model_path"])
    else:
        for task_name, task in tasks.items():
            model.add_task(task)

    emmental_learner = EmmentalLearner()
    emmental_learner.learn(model, dataloaders)

    scores = model.score(dataloaders)
    logger.info(f"Metrics: {scores}")
    write_to_json_file(f"{Meta.log_path}/metrics.txt", scores)

    if args.checkpointing:
        logger.info(
            f"Best metrics: "
            f"{emmental_learner.logging_manager.checkpointer.best_metric_dict}"
        )
        write_to_file(
            f"{Meta.log_path}/best_metrics.txt",
            emmental_learner.logging_manager.checkpointer.best_metric_dict,
        )
