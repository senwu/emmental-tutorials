import argparse
import logging
import os
import sys

import emmental
from emmental import Meta
from emmental.data import EmmentalDataLoader, EmmentalDataset
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_arg import parse_arg, parse_arg_to_config
from glue_tasks import get_gule_task
from preprocessor import preprocessor

logger = logging.getLogger(__name__)


def str2list(v):
    return [t.strip() for t in v.split(",")]


def write_to_file(file_name, value):
    if not isinstance(value, str):
        value = str(value)
    fout = open(os.path.join(Meta.log_path, file_name), "w")
    fout.write(value + "\n")
    fout.close()


def glue_scorer(metric_score_dict):
    metric_names = ["CoLA/GLUE/dev/matthews_corrcoef", "MNLI/GLUE/dev/accuracy", "MRPC/GLUE/dev/accuracy_f1", "QNLI/GLUE/dev/accuracy", "QQP/GLUE/dev/accuracy_f1", "RTE/GLUE/dev/accuracy", "SST-2/GLUE/dev/accuracy", "STS-B/GLUE/dev/pearson_spearman", "WNLI/GLUE/dev/accuracy"]
    
    total = 0.0
    cnt = 0

    for metric_name in metric_names:
        if metric_name not in metric_score_dict:
            continue
        else:
            total += metric_score_dict[metric_name]    
            cnt += 1

    return total / cnt


def add_application_args(parser):

    parser.add_argument("--task", type=str2list, required=True, help="GLUE tasks")

    parser.add_argument(
        "--data_dir", type=str, default="data", help="The path to GLUE dataset"
    )

    parser.add_argument(
        "--bert_model",
        type=str,
        default="bert-base-uncased",
        help="Which bert pretrained model to use",
    )

    parser.add_argument("--batch_size", type=int, default=16, help="batch size")

    parser.add_argument(
        "--max_data_samples", type=int, default=None, help="Maximum data samples to use"
    )

    parser.add_argument(
        "--max_sequence_length", type=int, default=200, help="Maximum sentence length"
    )


if __name__ == "__main__":
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        "GLUE Runner", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser = parse_arg(parser=parser)
    add_application_args(parser)

    args = parser.parse_args()
    config = parse_arg_to_config(args)

    emmental.init(config["meta_config"]["log_path"], config=config)

    cmd_msg = " ".join(sys.argv)
    logger.info(f"COMMAND: {cmd_msg}")
    write_to_file("cmd.txt", cmd_msg)

    logger.info(f"Config: {Meta.config}")
    write_to_file("config.txt", Meta.config)

    Meta.config["learner_config"]["global_evaluation_metric_dict"] = {"model/GLUE/dev/score": glue_scorer}
    datasets = {}

    for task_name in args.task:
        for split in ["train", "dev", "test"]:
            token_ids, token_segments, token_masks, labels = preprocessor(
                data_dir=args.data_dir,
                task_name=task_name,
                split=split,
                bert_model_name=args.bert_model,
                max_data_samples=args.max_data_samples,
                max_sequence_length=args.max_sequence_length,
            )
            X_dict = {
                "token_ids": token_ids,
                "token_segments": token_segments,
                "token_masks": token_masks,
            }
            Y_dict = {"labels": labels}

            if task_name not in datasets:
                datasets[task_name] = {}

            datasets[task_name][split] = EmmentalDataset(
                name="GLUE", X_dict=X_dict, Y_dict=Y_dict
            )

            logger.info(f"Loaded {split} for {task_name}.")

    dataloaders = []

    for task_name in args.task:
        for split in ["train", "dev", "test"]:
            dataloaders.append(
                EmmentalDataLoader(
                    task_to_label_dict={task_name: "labels"},
                    dataset=datasets[task_name][split],
                    split=split,
                    batch_size=args.batch_size,
                    shuffle=True if split == "train" else False,
                )
            )
            logger.info(f"Built dataloader for {task_name} {split} set.")

    tasks = get_gule_task(args.task, args.bert_model)

    mtl_model = EmmentalModel(name="GLUE_multi_task")

    if Meta.config["model_config"]["model_path"]:
        mtl_model.load(Meta.config["model_config"]["model_path"])
    else:
        for task_name, task in tasks.items():
            mtl_model.add_task(task)

    emmental_learner = EmmentalLearner()

    emmental_learner.learn(mtl_model, dataloaders)

    scores = mtl_model.score(dataloaders)
    logger.info(f"Metrics: {scores}")
    write_to_file("metrics.txt", scores)
    logger.info(f"Best metrics: {emmental_learner.logging_manager.checkpointer.best_metric_dict}")
    write_to_file("best_metrics.txt", emmental_learner.logging_manager.checkpointer.best_metric_dict)
