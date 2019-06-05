from functools import wraps
import logging
from types import SimpleNamespace

import numpy as np
import torch

logger = logging.getLogger(__name__)


class slicing_function():
    """
    When wrapped with this decorator, slicing functions only need to return an indicator
    for whether an individual example (bundle of attributes) belongs in that slice.
    Iterating through the dataset, making the pred array (and masking), etc. are all 
    handled automatically.
    """
    def __init__(self, fields=[]):
        self.fields = fields
    
    def __call__(self, f):
        @wraps(f)
        def wrapped_f(dataset):
            inds = []
            preds = []
            for idx in range(len(dataset)):
                example = SimpleNamespace(**{field: dataset.X_dict[field][idx] 
                                             for field in self.fields})
                ind = f(example)
                if ind == 0:
                    pred = 0
                else:
                    pred = dataset.Y_dict["labels"][idx]
                inds.append(ind)
                preds.append(pred)
            inds = torch.from_numpy(np.array(inds)).view(-1)
            preds = torch.from_numpy(np.array(preds)).view(-1)
            logger.info(f"Total {int((inds != 0).sum())} / {len(dataset)} examples are "
                        f"in slice {f.__name__}")
            return inds, preds
        return wrapped_f