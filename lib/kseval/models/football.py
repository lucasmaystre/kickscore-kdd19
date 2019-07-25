import abc
import itertools
import kickscore as ks
import numpy as np

from .base import PredictiveModel, KickScoreModel, iterate_dataset, YEAR
from .. import baselines

from datetime import datetime, timezone
from math import log


DATASET = "kdd-football.txt"


class TernaryModel(KickScoreModel, metaclass=abc.ABCMeta):

    dataset = DATASET

    def __init__(self, margin, obs_type="probit",
            method="ep", max_iter=500, lr=1.0):
        ks_model = ks.TernaryModel(margin=margin, obs_type=obs_type)
        fit_params = {"method": method, "max_iter": max_iter, "lr": lr}
        super().__init__(ks_model, fit_params)

    @abc.abstractmethod
    def make_features(self, team1, team2, neutral, bonus):
        """Make feature vectors for the two teams."""

    def observe(self, *, t, team1, team2, score1, score2, neutral, bonus):
        feats1, feats2 = self.make_features(team1, team2, neutral, bonus)
        # Add observation.
        if score1 > score2:
            self.ks_model.observe(feats1, feats2, t=t)
        elif score1 < score2:
            self.ks_model.observe(feats2, feats1, t=t)
        else:
            self.ks_model.observe(feats1, feats2, t=t, tie=True)

    def evaluate_obs(self, *, t, team1, team2, score1, score2, neutral, bonus):
        feats1, feats2 = self.make_features(team1, team2, neutral, bonus)
        probs = self.ks_model.probabilities(feats1, feats2, t=t)
        if score1 > score2:
            outcome = 0
        elif score1 == score2:
            outcome = 1
        else:
            outcome = 2
        log_loss = -log(probs[outcome])
        accuracy = 1.0 if outcome == np.argmax(probs) else 0.0
        return log_loss, accuracy


class ConstantModel(TernaryModel):

    def __init__(self, *, margin, cvar, method="ep", obs_type="probit"):
        super().__init__(margin, obs_type=obs_type, method=method)
        self._kern = ks.kernel.Constant(var=cvar)

    def make_features(self, team1, team2, neutral, bonus):
        # Add items if needed.
        if team1 not in self.ks_model.item:
            self.ks_model.add_item(team1, kernel=self._kern)
        if team2 not in self.ks_model.item:
            self.ks_model.add_item(team2, kernel=self._kern)
        return [team1], [team2]


class DynamicModel(TernaryModel):

    def __init__(self, *, nu, margin, cvar, dvar, lscale,
            method="ep", obs_type="probit"):
        super().__init__(margin, obs_type=obs_type, method=method)
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

    def make_features(self, team1, team2, neutral, bonus):
        # Add items if needed.
        if team1 not in self.ks_model.item:
            self.ks_model.add_item(team1, kernel=self._kern)
        if team2 not in self.ks_model.item:
            self.ks_model.add_item(team2, kernel=self._kern)
        return [team1], [team2]


class WienerModel(TernaryModel):

    def __init__(self, *, margin, cvar, wvar):
        super().__init__(margin, obs_type="probit", method="ep")
        WienerModel.init_items(self.ks_model, cvar, wvar)

    @staticmethod
    def init_items(ks_model, cvar, wvar):
        for obs in iterate_dataset(DATASET):
            for team in (obs["team1"], obs["team2"]):
                if team not in ks_model.item:
                    ks_model.add_item(team, kernel=ks.kernel.Wiener(
                            var=wvar/YEAR, t0=obs["t"], var_t0=cvar))

    def make_features(self, team1, team2, neutral, bonus):
        return [team1], [team2]


class HomeAdvantageModel(TernaryModel):

    def __init__(self, *, margin, cvar, dvar, lscale, havar):
        super().__init__(margin, obs_type="probit", method="ep")
        self._kern = (ks.kernel.Constant(var=cvar)
                + ks.kernel.Exponential(var=dvar, lscale=(lscale * YEAR)))
        self.ks_model.add_item("home-advantage",
                kernel=ks.kernel.Constant(var=havar))

    def make_features(self, team1, team2, neutral, bonus):
        # Add items if needed.
        if team1 not in self.ks_model.item:
            self.ks_model.add_item(team1, kernel=self._kern)
        if team2 not in self.ks_model.item:
            self.ks_model.add_item(team2, kernel=self._kern)
        # Account for the home-advantage if needed.
        if neutral:
            return [team1], [team2]
        else:
            return [team1, "home-advantage"], [team2]


class TTTModel(KickScoreModel):

    dataset = DATASET

    def __init__(self, *, margin, cvar, wvar):
        ks_model = ks.TernaryModel(margin=margin, obs_type="probit")
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

    def observe(self, *, t, team1, team2, score1, score2, neutral, bonus):
        year = datetime(datetime.fromtimestamp(
                t, timezone.utc).year, 1, 1).timestamp()
        if score1 > score2:
            self.ks_model.observe([team1], [team2], t=year)
        elif score1 < score2:
            self.ks_model.observe([team2], [team1], t=year)
        else:
            self.ks_model.observe([team1], [team2], t=year, tie=True)

    def evaluate_obs(self, *, t, team1, team2, score1, score2, neutral, bonus):
        year = datetime(datetime.fromtimestamp(
                t, timezone.utc).year, 1, 1).timestamp()
        probs = self.ks_model.probabilities([team1], [team2], t=year)
        if score1 > score2:
            outcome = 0
        elif score1 == score2:
            outcome = 1
        else:
            outcome = 2
        log_loss = -log(probs[outcome])
        accuracy = 1.0 if outcome == np.argmax(probs) else 0.0
        return log_loss, accuracy


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
            p1, pt, p2 = self.bl_model.predict(obs["team1"], obs["team2"])
            if obs["score1"] > obs["score2"]:
                self._loglike += log(p1)
                self.bl_model.observe(obs["team1"], obs["team2"])
            elif obs["score1"] == obs["score2"]:
                self._loglike += log(pt)
                self.bl_model.observe(obs["team1"], obs["team2"], tie=True)
            else:  # obs["score1"] < obs["score2"]:
                self._loglike += log(p2)
                self.bl_model.observe(obs["team2"], obs["team1"])

    def evaluate(self, *, begin):
        begin_ts = int(begin.timestamp())
        log_loss = 0
        accuracy = 0
        n_obs = 0
        for obs in iterate_dataset(DATASET):
            probs = self.bl_model.predict(obs["team1"], obs["team2"])
            if obs["t"] < begin_ts:
                continue
            if obs["score1"] > obs["score2"]:
                outcome = 0
                self.bl_model.observe(obs["team1"], obs["team2"])
            elif obs["score1"] == obs["score2"]:
                outcome = 1
                self.bl_model.observe(obs["team1"], obs["team2"], tie=True)
            else:  # obs["score1"] < obs["score2"]:
                outcome = 2
                self.bl_model.observe(obs["team2"], obs["team1"])
            log_loss += -log(probs[outcome])
            accuracy += 1.0 if np.argmax(probs) == outcome else 0.0
            n_obs += 1
        return n_obs, log_loss, accuracy

    @classmethod
    def get_dates(cls, begin=None):
        raise NotImplementedError()

    @property
    def log_likelihood(self):
        return self._loglike


class EloModel(BaselineModel):

    def __init__(self, *, margin, lr):
        super().__init__(baselines.EloModel(margin=margin, lr=lr))


class TrueSkillModel(BaselineModel):

    def __init__(self, *, margin, sigma, tau):
        super().__init__(baselines.TrueSkillModel(
                margin=margin, sigma=sigma, tau=tau))


class DifferenceModel(KickScoreModel):

    dataset = DATASET

    def __init__(self, ovar, cvar, dvar, lscale, threshold=0.5):
        ks_model = ks.DifferenceModel(var=ovar)
        fit_params = {"method": "ep", "max_iter": 500, "lr": 1.0}
        self._kern = (ks.kernel.Constant(var=cvar)
                + ks.kernel.Exponential(var=dvar, lscale=(lscale * YEAR)))
        super().__init__(ks_model, fit_params)
        self._thresh = threshold

    def observe(self, *, t, team1, team2, score1, score2, neutral, bonus):
        for team in (team1, team2):
            if team not in self.ks_model.item:
                self.ks_model.add_item(team, kernel=self._kern)
        self.ks_model.observe([team1], [team2], diff=(score1 - score2), t=t)

    def evaluate_obs(self, *, t, team1, team2, score1, score2, neutral, bonus):
        for team in (team1, team2):
            if team not in self.ks_model.item:
                self.ks_model.add_item(team, kernel=self._kern)
        p_win, _ = self.ks_model.probabilities(
                [team1], [team2], threshold=self._thresh, t=t)
        _, p_los = self.ks_model.probabilities(
                [team1], [team2], threshold=-self._thresh, t=t)
        p_tie = 1 - p_win - p_los
        if score1 > score2:
            return -log(p_win), 1.0 if p_win > max(p_tie, p_los) else 0.0
        elif score1 == score2:
            return -log(p_tie), 1.0 if p_tie > max(p_win, p_los) else 0.0
        else:
            return -log(p_los), 1.0 if p_los > max(p_win, p_tie) else 0.0


class CountModel(KickScoreModel):

    dataset = DATASET

    def __init__(self, cvar, dvar, lscale):
        ks_model = ks.CountModel()
        fit_params = {"method": "ep", "max_iter": 500, "lr": 0.8}
        self._kern = (ks.kernel.Constant(var=cvar)
                + ks.kernel.Exponential(var=dvar, lscale=(lscale * YEAR)))
        super().__init__(ks_model, fit_params)

    def observe(self, *, t, team1, team2, score1, score2, neutral, bonus):
        for team, pos in itertools.product((team1, team2), ("off", "def")):
            item = "{}-{}".format(team, pos)
            if item not in self.ks_model.item:
                self.ks_model.add_item(item, kernel=self._kern)
        self.ks_model.observe([team1+"-off"], [team2+"-def"], count=score1, t=t)
        self.ks_model.observe([team2+"-off"], [team1+"-def"], count=score2, t=t)

    def evaluate_obs(self, *, t, team1, team2, score1, score2, neutral, bonus):
        for team, pos in itertools.product((team1, team2), ("off", "def")):
            item = "{}-{}".format(team, pos)
            if item not in self.ks_model.item:
                self.ks_model.add_item(item, kernel=self._kern)
        ps1 = self.ks_model.probabilities([team1+"-off"], [team2+"-def"], t=t)
        ps2 = self.ks_model.probabilities([team2+"-off"], [team1+"-def"], t=t)
        p_win = 0.0
        p_tie = 0.0
        p2_lt = 0.0
        for k, (p1, p2) in enumerate(zip(ps1, ps2)):
            # 50% of ties are awarded to team 1.
            p_win += p1 * p2_lt
            p_tie += p1 * p2
            p2_lt += p2
        p_win += sum(ps1[k+1:])
        p_los = 1.0 - p_win - p_tie
        if score1 > score2:
            return -log(p_win), 1.0 if p_win > max(p_tie, p_los) else 0.0
        elif score1 == score2:
            return -log(p_tie), 1.0 if p_tie > max(p_win, p_los) else 0.0
        else:
            return -log(p_los), 1.0 if p_los > max(p_win, p_tie) else 0.0
