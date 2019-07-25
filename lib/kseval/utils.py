import os.path

from datetime import datetime, timezone


def data_path(fname):
    return os.path.join(os.environ["KSEVAL_DATASETS"], fname)


def parse_config(string):
    config = dict()
    for pair in string.split(","):
        k, v = pair.split("=")
        try:
            config[k] = int(v)
        except ValueError:
            try:
                config[k] = float(v)
            except ValueError:
                config[k] = v
    return config


def format_config(config):
    elems = list()
    for key, val in config.items():
        if isinstance(val, float):
            elems.append("{}={:.5f}".format(key, val))
        else:
            elems.append("{}={}".format(key, val))
    return ",".join(elems)


def parse_date(string):
    return datetime.strptime(string, "%Y-%m-%d").replace(tzinfo=timezone.utc)
