import json
import logging
import os
import re

import numpy as np
import torch
from task_config import SuperGLUE_LABEL_MAPPING, SuperGLUE_TASK_SPLIT_MAPPING
from tokenizer import get_tokenizer

from emmental.data import EmmentalDataLoader, EmmentalDataset

logger = logging.getLogger(__name__)


def parse_MultiRC_jsonl(
    jsonl_path, tokenizer, uid, max_data_samples, max_sequence_length
):

    task_name = "MultiRC"

    logger.info(f"Loading data from {jsonl_path}.")
    rows = [json.loads(row) for row in open(jsonl_path, encoding="utf-8")]
    #     logger.info(f"Sample 1: {rows[0]}")

    # Truncate to max_data_samples
    if max_data_samples:
        rows = rows[:max_data_samples]
        logger.info(f"Truncating to {max_data_samples} samples.")

    # unique ids
    uids = []
    # paragraph ids
    pids = []
    # question ids
    qids = []
    # answer ids
    aids = []

    # paragraph text
    paras = []
    # question text
    questions = []
    # answer text
    answers = []
    # labels
    labels = []

    bert_token_ids = []
    bert_token_masks = []
    bert_token_segments = []

    # Check the maximum token length
    max_len = -1

    for row in rows:
        # each example has a paragraph field -> (text, questions)
        # text is the paragraph, which requires some preprocessing
        # questions is a list of questions,
        # has fields (question, sentences_used, answers)
        pid = row["idx"]
        #         import pdb; pdb.set_trace()
        para = re.sub(
            "<b>Sent .{1,2}: </b>", "", row["paragraph"]["text"].replace("<br>", " ")
        )
        para_token = tokenizer.tokenize(para)[: max_sequence_length - 2]

        for ques in row["paragraph"]["questions"]:
            qid = ques["idx"]
            question = ques["question"]
            question_token = tokenizer.tokenize(question)[: max_sequence_length - 2]

            for ans in ques["answers"]:
                aid = ans["idx"]
                answer = ans["text"]
                answer_token = tokenizer.tokenize(answer)[: max_sequence_length - 2]

                # Generate tokens
                tokens = (
                    ["[CLS]"]
                    + para_token
                    + ["[SEP]"]
                    + question_token
                    + answer_token
                    + ["[SEP]"]
                )
                # No token segments
                token_segments = [0] * (len(para_token) + 2) + [0] * (
                    len(question_token) + len(answer_token) + 1
                )
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                token_masks = [1] * len(token_ids)

                if len(tokens) > max_len:
                    max_len = len(tokens)

                # Add to list
                paras.append(para)
                questions.append(question)
                answers.append(answer)

                label = ans["isAnswer"] if "isAnswer" in ans else False
                labels.append(SuperGLUE_LABEL_MAPPING[task_name][label])

                pids.append(pid)
                qids.append(qid)
                aids.append(aid)

                uids.append(f"{pid}%%{qid}%%{aid}")

                bert_token_ids.append(torch.LongTensor(token_ids))
                bert_token_masks.append(torch.LongTensor(token_masks))
                bert_token_segments.append(torch.LongTensor(token_segments))

    labels = torch.from_numpy(np.array(labels))

    logger.info(f"Max token len {max_len}")

    return EmmentalDataset(
        name="SuperGLUE",
        uid=uid,
        X_dict={
            "uids": uids,
            "pids": pids,
            "qids": qids,
            "aids": aids,
            "paras": paras,
            "questions": questions,
            "answers": answers,
            "token_ids": bert_token_ids,
            "token_masks": bert_token_masks,
            "token_segments": bert_token_segments,
        },
        Y_dict={"labels": labels},
    )


def get_dataloaders(
    data_dir,
    task_name="MultiRC",
    splits=["train", "val", "test"],
    max_data_samples=None,
    max_sequence_length=128,
    tokenizer_name="bert-base-uncased",
    batch_size=16,
    uid="uids",
):
    """Load data and return dataloaders"""

    dataloaders = {}

    tokenizer = get_tokenizer(tokenizer_name)

    for split in splits:
        jsonl_path = os.path.join(
            data_dir, task_name, SuperGLUE_TASK_SPLIT_MAPPING[task_name][split]
        )
        dataset = parse_MultiRC_jsonl(
            jsonl_path, tokenizer, uid, max_data_samples, max_sequence_length
        )

        dataloaders[split] = EmmentalDataLoader(
            task_to_label_dict={task_name: "labels"},
            dataset=dataset,
            split=split,
            batch_size=batch_size,
            shuffle=split == "train",
        )
        logger.info(f"Loaded {split} for {task_name} with {len(dataset)} samples.")

    return dataloaders
