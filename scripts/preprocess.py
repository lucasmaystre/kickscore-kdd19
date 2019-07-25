#!/usr/bin/env python
import argparse
import itertools
import json
import numpy as np
import os.path
import random

from datetime import datetime


# Initialize random seed for reproducibility.
random.seed(42)


def parse_tennis(path, cutoff=datetime(2017, 12, 31)):
    """Process Jeff Sackman's ATP dataset.

    The repository can be obtained at:
    <https://github.com/JeffSackmann/tennis_atp>
    """
    years = np.arange(1991, cutoff.year + 1, dtype=int)
    prefixes = ["atp_matches", "atp_matches_futures", "atp_matches_qual_chall"]
    data = list()
    for pfx, year in itertools.product(prefixes, years):
        with open(
                os.path.join(path, "{}_{}.csv".format(pfx, year)),
                errors="replace") as f:
            next(f)  # First line is header.
            for line in f:
                x = line.strip().split(",")
                if len(x) == 1 or x[5] == "":
                    continue  # Get rid of double carriage returns.
                dt = datetime.strptime(x[5], "%Y%m%d")
                if dt <= cutoff:
                    data.append({
                        "t": int(dt.timestamp()),
                        "winner": x[10],
                        "loser": x[20],
                    })
    for datum in sorted(data, key=lambda x: x["t"]):
        print(json.dumps(datum))


def parse_basketball(path, cutoff=datetime(2018, 12, 31)):
    """Process `nba_elo.csv`.

    This file can be obtained at:
    <https://projects.fivethirtyeight.com/nba-model/nba_elo.csv>
    """
    with open(path) as f:
        next(f)  # First line is header.
        for line in f:
            x = line.strip().split(",")
            dt = datetime.strptime(x[0], "%Y-%m-%d")
            if dt <= cutoff:
                print(json.dumps({
                    "t": int(dt.timestamp()),
                    "team1": x[4],
                    "team2": x[5],
                    "score1": int(x[22]),
                    "score2": int(x[23]),
                }))


def parse_chess(path, full=True):
    """Process `chessbase-2018.txt`.

    This is the output of `process-pgn.go` applied to the ChessBase 2018 Big
    Database. Contact Lucas Maystre for more information.
    """
    begin = datetime(1950, 1, 1)
    end = datetime(1979, 12, 31)
    epoch = datetime(1970, 1, 1)
    with open(path) as f:
        for line in f:
            date, white, black, winner = line.strip().split("|")
            if ("BYE" in white or "BYE" in black
                    or white == "NN" or black == "NN"):
                continue
            dt = datetime.strptime(date, "%Y-%m-%d")
            if full or begin <= dt <= end:
                print(json.dumps({
                    # We need this trick to be able to work with dates < 1900.
                    "t": int((dt - epoch).total_seconds()),
                    "white": white,
                    "black": black,
                    "winner": winner
                }))


def parse_starcraft(path):
    """Process `WoL.txt` or `HotS.txt`.

    These datasets are available at: <https://github.com/csinpi/blade_chest>.
    """
    obs = list()
    with open(path) as f:
        n_players = int(next(f).strip().split(" ")[1])
        for i in range(n_players):
            next(f)
        n_games = int(next(f).strip().split(" ")[1])
        for i in range(n_games):
            players, outcome = next(f).strip().split(" ")
            p1, p2 = map(int, players.split(":"))
            if outcome == "0:1":
                obs.append((p2, p1))
            elif outcome == "1:0":
                obs.append((p1, p2))
            else:
                raise RuntimeError("unknown outcome: '{}'".format(outcome))
    random.shuffle(obs)
    for winner, loser in obs:
        print(json.dumps({
            "winner": winner,
            "loser": loser,
            "t": 0,
        }))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=(
            "tennis", "basketball", "chess", "starcraft"))
    parser.add_argument("path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.which == "tennis":
        parse_tennis(args.path)
    elif args.which == "basketball":
        parse_basketball(args.path)
    elif args.which == "chess":
        parse_chess(args.path)
    else:  # args.which == "starcraft":
        parse_starcraft(args.path)
