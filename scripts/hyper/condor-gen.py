#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np

from kseval.utils import parse_date, format_config


HEADER = """
# Check condor primer (standard / vanilla).
Universe = vanilla

OutputDir = {output}

# Absolute path to executable (not relative to InitialDir).
Executable = {executable}

Error  = $(OutputDir)/err.$(Process)
Log    = $(OutputDir)/log.$(Process)
Output = $(OutputDir)/out.$(Process)

# This is to be turned on.
GetEnv = true

# IMPORTANT!!!! Otherwise you get screwed!
notification = Never

# Require NFS and `ephemeral` DFS.
requirements = nfshome_storage==true && ephemeral_storage==true

# Additional directives.
should_transfer_files = YES
transfer_input_files = condor-run.py
environment = KSEVAL_DATASETS={kseval_datasets}

### END OF HEADER"""

TEMPLATE_CUTOFF = """
Arguments = condor-run.py {cls} {config} --cutoff {cutoff:%Y-%m-%d}
Queue 1"""

TEMPLATE_NO_CUTOFF = """
Arguments = condor-run.py {cls} {config}
Queue 1"""


def sample_configs(path, n):
    with open(path) as f:
        info = json.load(f)
    keys = list()
    vals = list()
    for name, (kind, args) in info.items():
        keys.append(name)
        if kind == "lin":
            vals.append(np.random.uniform(
                    low=args["start"], high=args["stop"], size=n))
        elif kind == "geom":
            vals.append(np.exp(np.random.uniform(
                    low=np.log(args["start"]), high=np.log(args["stop"]),
                    size=n)))
        elif kind == "fixed":
            vals.append([args for _ in range(n)])
        else:
            raise RuntimeError("unknown type: '{}'".format(kind))
    for config in zip(*vals):
        yield dict(zip(keys, config))


def main(n, cls, range_path, out, cutoff):
    print(HEADER.format(
            executable=os.environ["CONDOR_PYTHON"],
            kseval_datasets=os.environ["KSEVAL_DATASETS"],
            output=os.path.abspath(out)))
    if cutoff is not None:
        template = TEMPLATE_CUTOFF
    else:
        template = TEMPLATE_NO_CUTOFF
    for config in sample_configs(range_path, n):
        print(template.format(
                cls=cls, config=format_config(config), cutoff=cutoff))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("model_class")
    parser.add_argument("model_range")
    parser.add_argument("output_folder")
    parser.add_argument("--cutoff", type=parse_date)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.n, args.model_class, args.model_range,
            args.output_folder, args.cutoff)
