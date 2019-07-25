#!/usr/bin/env python
import argparse
import json
import kseval.models

from datetime import timedelta
from kseval.utils import parse_date
from operator import attrgetter


def main(date, model_class, config_path):
    cls = attrgetter(model_class)(kseval.models)
    with open(config_path) as f:
        kwargs = json.load(f)
    model = cls(**kwargs)
    converged = model.fit(cutoff=date)
    n_obs, log_loss, accuracy = model.evaluate(
            begin=date, end=date+timedelta(days=1))
    print(json.dumps({
        "n_obs": n_obs,
        "log_loss": log_loss,
        "accuracy": accuracy,
        "date": "{:%Y-%m-%d}".format(date),
    }))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("date", type=parse_date)
    parser.add_argument("model_class")
    parser.add_argument("config_path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.date, args.model_class, args.config_path)
