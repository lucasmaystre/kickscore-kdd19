import abc
import itertools
import kickscore as ks
import numpy as np

from .base import PredictiveModel, KickScoreModel, iterate_dataset, YEAR
from .. import baselines

from datetime import datetime, timezone
from math import log


DATASET = "kdd-basketball.txt"


class BinaryModel(KickScoreModel, metaclass=abc.ABCMeta):

    dataset = DATASET

    def __init__(self, obs_type="probit", method="ep", max_iter=500, lr=1.0):
        ks_model = ks.BinaryModel(obs_type=obs_type)
        fit_params = {"method": method, "max_iter": max_iter, "lr": lr}
        super().__init__(ks_model, fit_params)

    @abc.abstractmethod
    def make_features(self, team1, team2):
        """Make feature vectors for the two teams."""

    def observe(self, *, t, team1, team2, score1, score2):
        feats1, feats2 = self.make_features(team1, team2)
        # Add observation.
        if score1 > score2:
            self.ks_model.observe(feats1, feats2, t=t)
        else:  # score1 < score2:
            self.ks_model.observe(feats2, feats1, t=t)

    def evaluate_obs(self, *, t, team1, team2, score1, score2):
        feats1, feats2 = self.make_features(team1, team2)
        p1, p2 = self.ks_model.probabilities(feats1, feats2, t=t)
        if score1 > score2:
            return -log(p1), 1.0 if p1 > p2 else 0.0
        else:  # score1 < score2:
            return -log(p2), 1.0 if p1 < p2 else 0.0


class ConstantModel(BinaryModel):

    def __init__(self, *, cvar, method="ep", obs_type="probit"):
        super().__init__(obs_type=obs_type, method=method)
        self._kern = ks.kernel.Constant(var=cvar)

    def make_features(self, team1, team2):
        # Add items if needed.
        if team1 not in self.ks_model.item:
            self.ks_model.add_item(team1, kernel=self._kern)
        if team2 not in self.ks_model.item:
            self.ks_model.add_item(team2, kernel=self._kern)
        return [team1], [team2]


class PiecewiseConstantModel(BinaryModel):

    def __init__(self, *, cvar, pvar, method="ep", obs_type="probit"):
        super().__init__(obs_type=obs_type, method=method)
        bounds = list()
        for year in range(1945, 2020):
            bounds.append(
                    datetime(year, 8, 1, tzinfo=timezone.utc).timestamp())
        self._kern = (ks.kernel.Constant(var=cvar)
                + ks.kernel.PiecewiseConstant(var=pvar, bounds=bounds))

    def make_features(self, team1, team2):
        # Add items if needed.
        if team1 not in self.ks_model.item:
            self.ks_model.add_item(team1, kernel=self._kern)
        if team2 not in self.ks_model.item:
            self.ks_model.add_item(team2, kernel=self._kern)
        return [team1], [team2]


class DynamicModel(BinaryModel):

    def __init__(self, *, nu, cvar, dvar, lscale,
            method="ep", obs_type="probit"):
        super().__init__(obs_type=obs_type, method=method)
        if nu == "1/2":
            dkern = ks.kernel.Exponential
        elif nu == "3/2":
            dkern = ks.kernel.Matern32
        elif nu == "5/2":
            dkern = ks.kernel.Matern52
        else:
            raise ValueError("invalid value for `nu`: {}".format(nu))
        self._kern = (ks.kernel.Constant(var=cvar)
                + dkern(var=dvar, lscale=(lscale * YEAR)))

    def make_features(self, team1, team2):
        # Add items if needed.
        if team1 not in self.ks_model.item:
            self.ks_model.add_item(team1, kernel=self._kern)
        if team2 not in self.ks_model.item:
            self.ks_model.add_item(team2, kernel=self._kern)
        return [team1], [team2]


class PiecewiseDynamicModel(BinaryModel):

    def __init__(self, *, nu, pvar, dvar, lscale,
            method="ep", obs_type="probit"):
        super().__init__(obs_type=obs_type, method=method)
        if nu == "1/2":
            dkern = ks.kernel.Exponential
        elif nu == "3/2":
            dkern = ks.kernel.Matern32
        elif nu == "5/2":
            dkern = ks.kernel.Matern52
        else:
            raise ValueError("invalid value for `nu`: {}".format(nu))
        bounds = list()
        for year in range(1945, 2020):
            bounds.append(
                    datetime(year, 8, 1, tzinfo=timezone.utc).timestamp())
        self._kern = (dkern(var=dvar, lscale=(lscale * YEAR))
                + ks.kernel.PiecewiseConstant(var=pvar, bounds=bounds))

    def make_features(self, team1, team2):
        # Add items if needed.
        if team1 not in self.ks_model.item:
            self.ks_model.add_item(team1, kernel=self._kern)
        if team2 not in self.ks_model.item:
            self.ks_model.add_item(team2, kernel=self._kern)
        return [team1], [team2]


class WienerModel(BinaryModel):

    def __init__(self, *, cvar, wvar):
        super().__init__(obs_type="probit", method="ep")
        WienerModel.init_items(self.ks_model, cvar, wvar)

    @staticmethod
    def init_items(ks_model, cvar, wvar):
        for obs in iterate_dataset(DATASET):
            for team in (obs["team1"], obs["team2"]):
                if team not in ks_model.item:
                    ks_model.add_item(team, kernel=ks.kernel.Wiener(
                            var=wvar/YEAR, t0=obs["t"], var_t0=cvar))

    def make_features(self, team1, team2):
        return [team1], [team2]


class TTTModel(KickScoreModel):

    dataset = DATASET

    def __init__(self, *, cvar, wvar):
        ks_model = ks.BinaryModel(obs_type="probit")
        TTTModel.init_items(ks_model, cvar, wvar)
        fit_params = {"method": "ep", "max_iter": 500, "lr": 1.0}
        super().__init__(ks_model, fit_params)

    @staticmethod
    def init_items(ks_model, cvar, wvar):
        for obs in iterate_dataset(DATASET):
            for team in (obs["team1"], obs["team2"]):
                if team not in ks_model.item:
                    year = datetime(datetime.fromtimestamp(
                            obs["t"], timezone.utc).year, 1, 1).timestamp()
                    ks_model.add_item(team, kernel=ks.kernel.Wiener(
                            var=wvar/YEAR, t0=year, var_t0=cvar))

    def observe(self, *, t, team1, team2, score1, score2):
        year = datetime(datetime.fromtimestamp(
                t, timezone.utc).year, 1, 1).timestamp()
        if score1 > score2:
            self.ks_model.observe([team1], [team2], t=year)
        else:  # score1 < score2:
            self.ks_model.observe([team2], [team1], t=year)

    def evaluate_obs(self, *, t, team1, team2, score1, score2):
        year = datetime(datetime.fromtimestamp(
                t, timezone.utc).year, 1, 1).timestamp()
        p1, p2 = self.ks_model.probabilities([team1], [team2], t=year)
        if score1 > score2:
            return -log(p1), 1.0 if p1 > p2 else 0.0
        else:  # score1 < score2:
            return -log(p2), 1.0 if p1 < p2 else 0.0


class BaselineModel(PredictiveModel):

    def __init__(self, bl_model):
        self.bl_model = bl_model
        self._loglike = 0  # Log-likelihood

    def fit(self, *, cutoff=None):
        if cutoff is not None:
            cutoff_ts = int(cutoff.timestamp())
        else:
            cutoff_ts = float("inf")
        self._loglike = 0
        for obs in iterate_dataset(DATASET):
            if obs["t"] >= cutoff_ts:
                break
            p1, _, p2 = self.bl_model.predict(obs["team1"], obs["team2"])
            if obs["score1"] > obs["score2"]:
                self._loglike += log(p1)
                self.bl_model.observe(obs["team1"], obs["team2"])
            else:  # obs["score1"] < obs["score2"]:
                self._loglike += log(p2)
                self.bl_model.observe(obs["team2"], obs["team1"])

    def evaluate(self, *, begin):
        begin_ts = int(begin.timestamp())
        log_loss = 0
        accuracy = 0
        n_obs = 0
        for obs in iterate_dataset(DATASET):
            p1, _, p2 = self.bl_model.predict(obs["team1"], obs["team2"])
            if obs["t"] < begin_ts:
                continue
            if obs["score1"] > obs["score2"]:
                log_loss += -log(p1)
                accuracy += 1.0 if p1 > p2 else 0.0
                self.bl_model.observe(obs["team1"], obs["team2"])
            else:  # obs["score1"] < obs["score2"]:
                log_loss += -log(p2)
                accuracy += 1.0 if p1 < p2 else 0.0
                self.bl_model.observe(obs["team2"], obs["team1"])
            n_obs += 1
        return n_obs, log_loss, accuracy

    @classmethod
    def get_dates(cls, begin=None):
        raise NotImplementedError()

    @property
    def log_likelihood(self):
        return self._loglike


class EloModel(BaselineModel):

    def __init__(self, *, lr):
        super().__init__(baselines.EloModel(margin=0, lr=lr))


class TrueSkillModel(BaselineModel):

    def __init__(self, *, sigma, tau):
        super().__init__(baselines.TrueSkillModel(
                margin=0, sigma=sigma, tau=tau))


class DifferenceModel(KickScoreModel):

    dataset = DATASET

    def __init__(self, ovar, cvar, dvar, lscale):
        ks_model = ks.DifferenceModel(var=ovar)
        fit_params = {"method": "ep", "max_iter": 500, "lr": 1.0}
        self._kern = (ks.kernel.Constant(var=cvar)
                + ks.kernel.Exponential(var=dvar, lscale=(lscale * YEAR)))
        super().__init__(ks_model, fit_params)

    def observe(self, *, t, team1, team2, score1, score2):
        for team in (team1, team2):
            if team not in self.ks_model.item:
                self.ks_model.add_item(team, kernel=self._kern)
        self.ks_model.observe([team1], [team2], diff=(score1 - score2), t=t)

    def evaluate_obs(self, *, t, team1, team2, score1, score2):
        for team in (team1, team2):
            if team not in self.ks_model.item:
                self.ks_model.add_item(team, kernel=self._kern)
        p_win, p_los = self.ks_model.probabilities([team1], [team2], t=t)
        if score1 > score2:
            return -log(p_win), 1.0 if p_win > p_los else 0.0
        else:
            return -log(p_los), 1.0 if p_los > p_win else 0.0


class CountModel(KickScoreModel):

    dataset = DATASET

    def __init__(self, cvar, dvar, lscale):
        ks_model = ks.CountModel()
        fit_params = {"method": "ep", "max_iter": 500, "lr": 0.8}
        self._kern = (ks.kernel.Constant(var=cvar)
                + ks.kernel.Exponential(var=dvar, lscale=(lscale * YEAR)))
        super().__init__(ks_model, fit_params)

    def observe(self, *, t, team1, team2, score1, score2):
        for team, pos in itertools.product((team1, team2), ("off", "def")):
            item = "{}-{}".format(team, pos)
            if item not in self.ks_model.item:
                self.ks_model.add_item(item, kernel=self._kern)
        self.ks_model.observe([team1+"-off"], [team2+"-def"], count=score1, t=t)
        self.ks_model.observe([team2+"-off"], [team1+"-def"], count=score2, t=t)

    def evaluate_obs(self, *, t, team1, team2, score1, score2):
        for team, pos in itertools.product((team1, team2), ("off", "def")):
            item = "{}-{}".format(team, pos)
            if item not in self.ks_model.item:
                self.ks_model.add_item(item, kernel=self._kern)
        ps1 = self.ks_model.probabilities([team1+"-off"], [team2+"-def"], t=t)
        ps2 = self.ks_model.probabilities([team2+"-off"], [team1+"-def"], t=t)
        p_win = 0.0
        p2_lt = 0.0
        for k, (p1, p2) in enumerate(zip(ps1, ps2)):
            # 50% of ties are awarded to team 1.
            p_win += p1 * (0.5 * p2 + p2_lt)
            p2_lt += p2
        p_win += sum(ps1[k+1:])
        if score1 > score2:
            return -log(p_win), 1.0 if p_win >= 0.5 else 0.0
        else:
            return -log(1 - p_win), 1.0 if p_win < 0.5 else 0.0
