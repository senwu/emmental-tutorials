"""
Script the calls `run.py` using fixed_args and search_space specified in <config>.json file.

Sample:
python search.py --max_search 2 --config /dfs/scratch0/vschen/emmental-tutorials/superglue/example_config.json
"""

import argparse
import datetime
import json
import os
import pprint
import random
import subprocess
from itertools import cycle, product

import numpy as np
from run import get_parser
from run import main as launch


def config_generator(search_space, max_search, rng, shuffle=True):
    """Generates config dicts from the given search space
    Args:
        search_space: (dict) A dictionary of parameters to search over.
            See note below for more details.
        max_search: (int) The maximum number of configurations to search.
            If max_search is None, do a full grid search of all discrete
                parameters, filling in range parameters as needed.
            Otherwise, do a full grid search of all discrete
                parameters and then cycle through again filling in new
                range parameters values; if there are no range parameters,
                stop after yielding the full cross product of parameters
                once.
        shuffle: (bool) If True, shuffle the order of generated configs
    Yields:
        configs: each config is a dict of parameter values based on the
            provided search space
    The search_space dictionary may consist of two types of parameters:
        --discrete: a discrete parameter is either a single value or a
            list of values. Use single values, for example, to override
            a default model parameter or set a flag such as 'verbose'=True.
        --range: a range parameter is a dict of the form:
            {'range': [<min>, <max>], 'scale': <scale>}
            where <min> and <max> are the min/max values to search between
            and scale is one of ['linear', 'log'] (defaulting to 'linear')
            representing the scale to use when searching the given range
    Example:
        search_space = {
            'verbose': True,                              # discrete
            'n_epochs': 100,                              # discrete
            'momentum': [0.0, 0.9, 0.99],                       # discrete
            'l2': {'range': [0.0001, 10]}                 # linear range
            'lr': {'range': [0.001, 1], 'scale': 'log'},  # log range
        }
        If max_search is None, this will return 3 configurations (enough to
            just cover the full cross-product of discrete values, filled
            in with sampled range values)
        Otherewise, this will return max_search configurations
            (cycling through the discrete value combinations multiple
            time if necessary)
    """

    def dict_product(d):
        keys = d.keys()
        for element in product(*d.values()):
            yield dict(zip(keys, element))

    def range_param_func(v):
        scale = v.get("scale", "linear")
        mini = min(v["range"])
        maxi = max(v["range"])
        if scale == "linear":
            func = lambda rand: mini + (maxi - mini) * rand
        elif scale == "log":
            mini = np.log(mini)
            maxi = np.log(maxi)
            func = lambda rand: np.exp(mini + (maxi - mini) * rand)
        else:
            raise ValueError(f"Unrecognized scale '{scale}' for " "parameter {k}")
        return func

    discretes = {}
    ranges = {}
    for k, v in search_space.items():
        if isinstance(v, dict):
            ranges[k] = range_param_func(v)
        elif isinstance(v, list):
            discretes[k] = v
        else:
            discretes[k] = [v]

    discrete_configs = list(dict_product(discretes))

    if shuffle:
        rng.shuffle(discrete_configs)

    # If there are range parameters and a non-None max_search, cycle
    # through the discrete_configs (with new range values) until
    # max_search is met
    if ranges and max_search:
        discrete_configs = cycle(discrete_configs)

    for i, config in enumerate(discrete_configs):
        # We may see the same config twice due to cycle
        config = config.copy()
        if max_search and i == max_search:
            break
        for k, v in ranges.items():
            config[k] = float(v(rng.random()))
        yield config


def main(args):
    search_config = json.load(open(args.config, "r"))
    assert "search_space" in search_config and "fixed_args" in search_config
    search_space = search_config["search_space"]
    fixed_args = search_config["fixed_args"]

    configs = config_generator(
        search_space, args.max_search, random.Random(args.search_seed), True
    )

    config_to_metrics = {}
    for search_conf in configs:
        full_conf = {}
        full_conf.update(search_conf)
        if any([k in full_conf for k in fixed_args.keys()]):
            raise ValueError("Fixed arg shows up in search space.")
        full_conf.update(fixed_args)
        if args.device:
            full_conf.update({"device": args.device})

        arg_list = []
        for k, v in full_conf.items():
            # make sure the double quotes are correctly formatted
            if isinstance(v, dict):
                v = json.dumps(v)
            arg_list.extend([f"--{k}", str(v)])

        # print command being run
        print("*" * 80)
        print("python run.py", " ".join(arg_list))
        print("*" * 80)
        parser = get_parser()
        launch_args = parser.parse_args(arg_list)
        metrics_path = launch(launch_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_search",
        type=int,
        default=None,
        help=(
            "Number of iterations to search (see tuner.config_generator). "
            "If None, searches all discrete combinations."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config .json with fixed_args and search_space fields",
    )
    parser.add_argument(
        "--device", type=str, help="Device to train on. See trainer kwargs for more."
    )
    parser.add_argument(
        "--search_seed",
        type=int,
        help="Seed for random search. Separate from run.py seed, specified in config",
        default=123,
    )
    args = parser.parse_args()
    main(args)
