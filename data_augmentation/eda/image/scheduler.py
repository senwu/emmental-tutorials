import logging
import random
from random import choices
from typing import Dict, Iterator, List, Tuple, Union

import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch import Tensor

from emmental.data import EmmentalDataLoader
from emmental.schedulers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class AugScheduler(Scheduler):
    r"""Generate batch generator from all dataloaders in round robin order for MTL
      training.

    Args:
      fillup(bool): Whether fillup to make all dataloader the same size.

    """

    def __init__(
        self, fillup: bool = False, augment_k: int = None, enlarge: int = 1
    ) -> None:
        super().__init__()
        self.fillup = fillup
        self.augment_k = augment_k
        self.enlarge = enlarge

        assert (
            self.augment_k is None or self.enlarge <= self.augment_k
        ), f"{self.enlarge} <= {self.augment_k}"

    def get_num_batches(self, dataloaders: List[EmmentalDataLoader]) -> int:
        r"""Get total number of batches per epoch.

        Args:
          dataloaders(list): List of dataloaders.

        Returns:
          int: Total number of batches per epoch.

        """

        batch_counts = [len(dataloader) for dataloader in dataloaders]
        if self.fillup:
            batch_counts = [max(batch_counts)] * len(dataloaders)

        for idx in range(len(dataloaders)):
            if dataloaders[idx].n_batches:
                batch_counts[idx] = dataloaders[idx].n_batches

        return sum(batch_counts)

    def get_batches(
        self, dataloaders: List[EmmentalDataLoader], model
    ) -> Iterator[
        Tuple[
            List[str],
            Dict[str, Union[Tensor, List[str]]],
            Dict[str, Tensor],
            Dict[str, str],
            str,
            str,
        ]
    ]:
        r"""Generate batch generator from all dataloaders in round robin order for
          one epoch.

        Args:
          dataloaders(list): List of dataloaders.

        Returns:
          genertor: A generator of all batches.

        """
        task_to_label_dicts = [
            dataloader.task_to_label_dict for dataloader in dataloaders
        ]
        uid_names = [dataloader.uid for dataloader in dataloaders]
        data_names = [dataloader.data_name for dataloader in dataloaders]
        splits = [dataloader.split for dataloader in dataloaders]
        data_loaders = [iter(dataloader) for dataloader in dataloaders]

        # Calc the batch size for each dataloader
        batch_counts = [len(dataloader) for dataloader in dataloaders]
        if self.fillup:
            batch_counts = [max(batch_counts)] * len(dataloaders)

        for idx in range(len(dataloaders)):
            if dataloaders[idx].n_batches:
                batch_counts[idx] = dataloaders[idx].n_batches

        dataloader_indexer = []
        for idx, count in enumerate(batch_counts):
            dataloader_indexer.extend([idx] * count)

        random.shuffle(dataloader_indexer)
        for data_loader_idx in dataloader_indexer:
            uid_name = uid_names[data_loader_idx]
            try:
                X_dict, Y_dict = next(data_loaders[data_loader_idx])
            except StopIteration:
                data_loaders[data_loader_idx] = iter(dataloaders[data_loader_idx])
                X_dict, Y_dict = next(data_loaders[data_loader_idx])

            if self.augment_k and self.augment_k > 1 and self.augment_k > self.enlarge:
                model.eval()
                with torch.no_grad():
                    uid_dict, loss_dict, prob_dict, gold_dict = model(
                        X_dict[uid_name],
                        X_dict,
                        Y_dict,
                        task_to_label_dicts[data_loader_idx],
                    )
                model.train()

                # Collect losses
                loss_dist = list(loss_dict.values())[0].detach().cpu().numpy()

                # row-based weighted sampling
                dist = normalize(
                    np.array(loss_dist).reshape(-1, self.augment_k), axis=1, norm="l1"
                )
                select_idx = np.vstack(
                    [
                        i * self.augment_k
                        + np.array(
                            choices(range(self.augment_k), dist[i], k=self.enlarge)
                        )
                        if max(dist[i]) > 0
                        else i * self.augment_k
                        + np.array(choices(range(self.augment_k), k=self.enlarge))
                        for i in range(dist.shape[0])
                    ]
                ).reshape(-1)

                X_new_dict = {"image": [], uid_name: []}
                Y_new_dict = {"labels": []}
                for idx in select_idx:
                    X_new_dict[uid_name].append(X_dict[uid_name][idx])
                X_new_dict["image"] = X_dict["image"][select_idx]
                Y_new_dict["labels"] = Y_dict["labels"][select_idx]

                X_dict = X_new_dict
                Y_dict = Y_new_dict
            yield X_dict[uid_name], X_dict, Y_dict, task_to_label_dicts[
                data_loader_idx
            ], data_names[data_loader_idx], splits[data_loader_idx]
