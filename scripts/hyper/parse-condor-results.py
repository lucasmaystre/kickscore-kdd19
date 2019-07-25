#!/usr/bin/env python
import argparse
import collections
import json
import numpy as np

from kseval.utils import format_config


SPECIAL_KEYS = set(["log_likelihood", "converged"])


def main(path, n_configs):
    data = list()
    vals = collections.defaultdict(set)
    with open(path) as f:
        for line in f:
            datum = json.loads(line.strip())
            if np.isnan(datum["log_likelihood"]):
                # Ignore entries where log-likelihood is NaN.
                continue
            data.append(datum)
            for k, v in datum.items():
                if k not in SPECIAL_KEYS:
                    vals[k].add(v)
    params = list(k for k in sorted(data[0].keys()) if k not in SPECIAL_KEYS)
    # Print the best configurations.
    print("Best configurations:")
    for config in sorted(
            data, key=lambda x: x["log_likelihood"], reverse=True)[:n_configs]:
        s = format_config({k: config[k] for k in params})
        print("{} -> {:.2f} (converged: {})".format(
                s, config["log_likelihood"], config.get("converged")))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--n-configs", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.path, args.n_configs)
