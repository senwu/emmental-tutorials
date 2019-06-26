import logging
from collections import defaultdict
from functools import wraps
from types import SimpleNamespace

import numpy as np
import torch
from emmental.data import EmmentalDataset, emmental_collate_fn
from emmental.utils.utils import list_to_tensor

logger = logging.getLogger(__name__)

# NOTE: This must match Meta.config["learner_config"]["ignore_index"]
IGNORE_INDEX = 0


class augmentation_function:
    """
    When wrapped with this decorator, augmentation functions only need to accept a 
    single (x_dict, y_dict) and return a new (x_dict, y_dict).
    If no new example should be made from this example, the AF should return (None, None).
    """

    # def __init__(self, name):
    #     self.fields = fields

    def __call__(self, f):
        @wraps(f)
        def wrapped_f(dataset):
            X_dict = defaultdict(list)
            Y_dict = defaultdict(list)
            examples = []
            for x_dict, y_dict in dataset:
                # TODO: Consider making sure aug_x_dict is not x_dict!
                aug_x_dict, aug_y_dict = f(x_dict, y_dict)
                if aug_x_dict is not None and aug_y_dict is not None:
                    examples.append((aug_x_dict, aug_y_dict))
            for x_dict, y_dict in examples:
                for k, v in x_dict.items():
                    X_dict[k].append(v)
                for k, v in y_dict.items():
                    Y_dict[k].append(v)
            for k, v in Y_dict.items():
                Y_dict[k] = list_to_tensor(v)
            # X_dict, Y_dict = emmental_collate_fn(examples)
            aug_dataset = EmmentalDataset(name=f.__name__, X_dict=X_dict, Y_dict=Y_dict)
            logger.info(
                f"Total {len(aug_dataset)} augmented examples were created "
                f"from AF {f.__name__}"
            )
            return aug_dataset

        return wrapped_f
