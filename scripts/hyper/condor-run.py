#!/usr/bin/env python
import argparse
import json
import kseval.models

from operator import attrgetter
from kseval.utils import parse_config, parse_date


def main(model_class, model_config, cutoff):
    cls = attrgetter(model_class)(kseval.models)
    model = cls(**model_config)
    converged = model.fit(cutoff=cutoff)
    print(json.dumps({
        **model_config,
        "log_likelihood": model.log_likelihood,
        "converged": converged,
        "cutoff": "{:%Y-%m-%d}".format(cutoff) if cutoff else None
    }))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_class")
    parser.add_argument("model_config", type=parse_config)
    parser.add_argument("--cutoff", type=parse_date)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.model_class, args.model_config, args.cutoff)
