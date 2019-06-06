import argparse
import logging
import sys
from functools import partial

import emmental
import models
import slicing
from dataloaders import get_dataloaders
from emmental import Meta
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_arg import parse_arg, parse_arg_to_config, str2bool
from utils import str2list, write_to_file

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
    parser.add_argument("--slices", type=str2bool, default=False, help="Whether to include slices")

    parser.add_argument(
        "--max_data_samples", type=int, default=None, help="Maximum data samples to use"
    )

    parser.add_argument(
        "--max_sequence_length", type=int, default=200, help="Maximum sentence length"
    )

    parser.add_argument(
        "--train", type=str2bool, default=True, help="Whether to train the model"
    )


if __name__ == "__main__":
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        "SuperGLUE Runner", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser = parse_arg(parser=parser)
    add_application_args(parser)

    args = parser.parse_args()
    config = parse_arg_to_config(args)

    # Initialize Emmental
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
    superglue_dataloaders = []
    superglue_tasks = []

    for task_name in args.task:
        dataloaders = get_dataloaders(
            data_dir=args.data_dir,
            task_name=task_name,
            splits=["train", "val", "test"],
            max_sequence_length=args.max_sequence_length,
            max_data_samples=args.max_data_samples,
            tokenizer_name=args.bert_model,
            batch_size=args.batch_size,
        )
        task = models.model[task_name](args.bert_model)
        if args.slices:
            slice_func_dict = slicing.slice_func_dict[task_name]
            dataloaders = slicing.add_slice_labels(task_name, dataloaders, slice_func_dict)
            tasks = slicing.add_slice_tasks(task_name, task, slice_func_dict)
        else:
            tasks = [task]

        superglue_dataloaders.extend(dataloaders)
        superglue_tasks.extend(tasks)

    # Build Emmental model
    superglue_model = EmmentalModel(name=f"SuperGLUE", tasks=superglue_tasks)

    # Load pretrained model if necessary
    if Meta.config["model_config"]["model_path"]:
        superglue_model.load(Meta.config["model_config"]["model_path"])

    # Training
    if args.train:
        emmental_learner = EmmentalLearner()
        emmental_learner.learn(superglue_model, superglue_dataloaders)

    scores = superglue_model.score(superglue_dataloaders)

    # Slice scoring
    for task_name in args.task:
        scorer = superglue_model.scorers[task_name]
        slice_func_dict = slicing.slice_func_dict[task_name]
        for dataloader in superglue_dataloaders:
            if dataloader.split == "test":
                continue
            pred_dict = superglue_model.predict(dataloader, return_preds=True)
            golds = pred_dict["golds"][task_name]
            probs = pred_dict["probs"][task_name]
            preds = pred_dict["preds"][task_name]
            for slice_name, slice_func in slice_func_dict.items():
                if "slice_base" in slice_name:
                    continue
                inds, _ = slice_func(dataloader.dataset)
                mask = (inds == 1).numpy().astype(bool)
                print(f"Scoring on {len(golds[mask])} examples")
                slice_scores = scorer.score(golds[mask], probs[mask], preds[mask])
                for metric_name, metric_value in slice_scores.items():
                    identifier = "/".join(
                        [f"{task_name}:{slice_name}", 
                         dataloader.data_name,
                         dataloader.split, 
                         metric_name]
                    )
                    scores[identifier] = metric_value

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
