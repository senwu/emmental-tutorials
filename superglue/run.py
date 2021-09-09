import argparse
import logging
import os
import sys
from functools import partial

import models
import slicing
from dataloaders import get_dataloaders
from submit import make_submission_file
from utils import str2list, write_to_file

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_arg import parse_arg, parse_arg_to_config, str2bool

logger = logging.getLogger(__name__)


def superglue_scorer(metric_score_dict, split="val"):
    metric_names = [
        f"CB/SuperGLUE/{split}/accuracy_macro_f1",
        f"COPA/SuperGLUE/{split}/accuracy",
        f"MultiRC/SuperGLUE/{split}/em_f1",
        f"RTE/SuperGLUE/{split}/accuracy",
        f"WiC/SuperGLUE/{split}/accuracy",
        f"WSC/SuperGLUE/{split}/accuracy",
    ]

    total = 0.0
    cnt = 0

    for metric_name in metric_names:
        if metric_name not in metric_score_dict:
            continue
        else:
            total += metric_score_dict[metric_name]
            cnt += 1

    return total / cnt if cnt > 0 else 0


def add_application_args(parser):

    parser.add_argument("--task", type=str2list, required=True, help="GLUE tasks")

    parser.add_argument(
        "--data_dir", type=str, default="data", help="The path to GLUE dataset"
    )

    parser.add_argument(
        "--bert_model",
        type=str,
        default="bert-large-cased",
        help="Which bert pretrained model to use",
    )

    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--slices", type=str2bool, default=False, help="Whether to include slices"
    )
    parser.add_argument(
        "--general_slices",
        type=str2bool,
        default=False,
        help="Whether to include general slices",
    )

    parser.add_argument(
        "--augmentations",
        type=str2bool,
        default=False,
        help="Whether to include augmentations",
    )

    parser.add_argument(
        "--slice_hidden_dim", type=int, default=1024, help="Slice hidden dimension size"
    )

    parser.add_argument(
        "--max_data_samples", type=int, default=None, help="Maximum data samples to use"
    )

    parser.add_argument(
        "--max_sequence_length", type=int, default=256, help="Maximum sentence length"
    )

    parser.add_argument(
        "--last_hidden_dropout_prob",
        type=float,
        default=0.0,
        help="Dropout on last layer of bert.",
    )

    parser.add_argument(
        "--train", type=str2bool, default=True, help="Whether to train the model"
    )


def get_parser():
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        "SuperGLUE Runner", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser = parse_arg(parser=parser)
    add_application_args(parser)
    return parser


def main(args):
    # Ensure that global state is fresh
    Meta.reset()

    # Initialize Emmental
    config = parse_arg_to_config(args)
    emmental.init(config["meta_config"]["log_path"], config=config)

    # Save command line argument into file
    cmd_msg = " ".join(sys.argv)
    logger.info(f"COMMAND: {cmd_msg}")
    write_to_file(Meta.log_path, "cmd.txt", cmd_msg)

    # Save Emmental config into file
    logger.info(f"Config: {Meta.config}")
    write_to_file(Meta.log_path, "config.txt", Meta.config)

    Meta.config["learner_config"]["global_evaluation_metric_dict"] = {
        f"model/SuperGLUE/{split}/score": partial(superglue_scorer, split=split)
        for split in ["val"]
    }

    # Construct dataloaders and tasks and load slices
    dataloaders = []
    tasks = []

    for task_name in args.task:
        task_dataloaders = get_dataloaders(
            data_dir=args.data_dir,
            task_name=task_name,
            splits=["train", "val", "test"],
            max_sequence_length=args.max_sequence_length,
            max_data_samples=args.max_data_samples,
            tokenizer_name=args.bert_model,
            batch_size=args.batch_size,
            augment=args.augmentations,
        )
        task = models.model[task_name](
            args.bert_model, last_hidden_dropout_prob=args.last_hidden_dropout_prob
        )
        if args.slices:
            logger.info("Initializing task-specific slices")
            slice_func_dict = slicing.slice_func_dict[task_name]
            # Include general purpose slices
            if args.general_slices:
                logger.info("Including general slices")
                slice_func_dict.update(slicing.slice_func_dict["general"])

            task_dataloaders = slicing.add_slice_labels(
                task_name, task_dataloaders, slice_func_dict
            )

            slice_tasks = slicing.add_slice_tasks(
                task_name, task, slice_func_dict, args.slice_hidden_dim
            )
            tasks.extend(slice_tasks)
        else:
            tasks.append(task)

        dataloaders.extend(task_dataloaders)

    # Build Emmental model
    model = EmmentalModel(name=f"SuperGLUE", tasks=tasks)

    # Load pretrained model if necessary
    if Meta.config["model_config"]["model_path"]:
        model.load(Meta.config["model_config"]["model_path"])

    # Training
    if args.train:
        emmental_learner = EmmentalLearner()
        emmental_learner.learn(model, dataloaders)

    # If model is slice-aware, slice scores will be calculated from slice heads
    # If model is not slice-aware, manually calculate performance on slices
    if not args.slices:
        slice_func_dict = {}
        slice_keys = args.task
        if args.general_slices:
            slice_keys.append("general")

        for k in slice_keys:
            slice_func_dict.update(slicing.slice_func_dict[k])

        scores = slicing.score_slices(model, dataloaders, args.task, slice_func_dict)
    else:
        scores = model.score(dataloaders)

    # Save metrics into file
    logger.info(f"Metrics: {scores}")
    write_to_file(Meta.log_path, "metrics.txt", scores)

    # Save best metrics into file
    if args.train:
        logger.info(
            f"Best metrics: "
            f"{emmental_learner.logging_manager.checkpointer.best_metric_dict}"
        )
        write_to_file(
            Meta.log_path,
            "best_metrics.txt",
            emmental_learner.logging_manager.checkpointer.best_metric_dict,
        )

    # Save submission file
    for task_name in args.task:
        dataloaders = [d for d in dataloaders if d.split == "test"]
        assert len(dataloaders) == 1
        filepath = os.path.join(Meta.log_path, f"{task_name}.jsonl")
        make_submission_file(model, dataloaders[0], task_name, filepath)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
