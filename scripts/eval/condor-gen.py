#!/usr/bin/env python3
import argparse
import json
import kseval.models
import os

from kseval.utils import parse_date
from operator import attrgetter


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
transfer_input_files = condor-run.py,{config}
environment = KSEVAL_DATASETS={kseval_datasets}

### END OF HEADER"""

TEMPLATE = """
Arguments = condor-run.py {date:%Y-%m-%d} {cls} {config}
Queue 1"""


def main(cls, config, out, begin):
    print(HEADER.format(
            executable=os.environ["CONDOR_PYTHON"],
            kseval_datasets=os.environ["KSEVAL_DATASETS"],
            output=os.path.abspath(out),
            config=os.path.abspath(config)))
    for date in attrgetter(cls)(kseval.models).get_dates(begin=begin):
        print(TEMPLATE.format(
                date=date, cls=cls, config=os.path.basename(config)))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_class")
    parser.add_argument("model_config")
    parser.add_argument("output_folder")
    parser.add_argument("--begin", type=parse_date)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.model_class, args.model_config, args.output_folder, args.begin)
