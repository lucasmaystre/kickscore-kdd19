import abc
import kickscore as ks
import numpy as np

from .base import PredictiveModel, KickScoreModel, iterate_dataset, YEAR
from .. import baselines

from datetime import datetime, timezone
from math import log


DATASET = "kdd-chess-small.txt"


class TernaryModel(KickScoreModel, metaclass=abc.ABCMeta):

    dataset = DATASET

    def __init__(self, margin, obs_type="probit",
            method="ep", max_iter=100, lr=1.0):
        ks_model = ks.TernaryModel(margin=margin, obs_type=obs_type)
        fit_params = {"method": method, "max_iter": max_iter, "lr": lr}
        super().__init__(ks_model, fit_params)

    @abc.abstractmethod
    def make_features(self, white, black):
        """Make feature vectors for the two players."""

    def observe(self, *, t, white, black, winner):
        feats1, feats2 = self.make_features(white, black)
        # Add observation.
        if winner == "white":
            self.ks_model.observe(feats1, feats2, t=t)
        elif winner == "black":
            self.ks_model.observe(feats2, feats1, t=t)
        else:  # winner = "tie":
            self.ks_model.observe(feats1, feats2, t=t, tie=True)

    def evaluate_obs(self, *, t, white, black, winner):
        feats1, feats2 = self.make_features(white, black)
        probs = self.ks_model.probabilities(feats1, feats2, t=t)
        if winner == "white":
            outcome = 0
        elif winner == "black":
            outcome = 2
        else:  # winner == "tie":
            outcome = 1
        log_loss = -log(probs[outcome])
        accuracy = 1.0 if outcome == np.argmax(probs) else 0.0
        return log_loss, accuracy


class ConstantModel(TernaryModel):

    def __init__(self, *, margin, cvar, method="ep", obs_type="probit"):
        super().__init__(margin, obs_type=obs_type, method=method)
        self._kern = ks.kernel.Constant(var=cvar)

    def make_features(self, white, black):
        # Add items if needed.
        for item in (white, black):
            if item not in self.ks_model.item:
                self.ks_model.add_item(item, kernel=self._kern)
        return [white], [black]


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

    def make_features(self, white, black):
        # Add items if needed.
        for item in (white, black):
            if item not in self.ks_model.item:
                self.ks_model.add_item(item, kernel=self._kern)
        return [white], [black]


class WienerModel(TernaryModel):

    def __init__(self, *, margin, cvar, wvar, method="ep"):
        super().__init__(margin, obs_type="probit", method=method)
        WienerModel.init_items(self.ks_model, cvar, wvar)

    @staticmethod
    def init_items(ks_model, cvar, wvar):
        for obs in iterate_dataset(DATASET):
            for item in (obs["white"], obs["black"]):
                if item not in ks_model.item:
                    ks_model.add_item(item, kernel=ks.kernel.Wiener(
                            var=wvar/YEAR, t0=obs["t"], var_t0=cvar))

    def make_features(self, white, black):
        return [white], [black]


class WhiteAdvantageModel(TernaryModel):

    def __init__(self, *, margin, cvar, dvar, lscale, havar):
        super().__init__(margin, obs_type="probit", method="ep")
        self._kern = (ks.kernel.Constant(var=cvar)
                + ks.kernel.Exponential(var=dvar, lscale=(lscale * YEAR)))
        self.ks_model.add_item("white-advantage",
                kernel=ks.kernel.Constant(var=havar))

    def make_features(self, white, black):
        # Add items if needed.
        for item in (white, black):
            if item not in self.ks_model.item:
                self.ks_model.add_item(item, kernel=self._kern)
        # Account for the white-advantage.
        return [white, "white-advantage"], [black]


class AffineDynamicModel(KickScoreModel):

    dataset = DATASET

    def __init__(self, *, margin, ovar, svar, dvar, lscale):
        ks_model = ks.TernaryModel(margin=margin, obs_type="probit")
        AffineDynamicModel.init_items(ks_model, ovar, svar, dvar, lscale)
        fit_params = {"method": "ep", "max_iter": 500, "lr": 1.0}
        super().__init__(ks_model, fit_params)

    @staticmethod
    def init_items(ks_model, ovar, svar, dvar, lscale):
        for obs in iterate_dataset(DATASET):
            for item in (obs["white"], obs["black"]):
                if item not in ks_model.item:
                    k = ks.kernel.Affine(
                            var_offset=ovar, var_slope=svar, t0=obs["t"]/YEAR)
                    k += ks.kernel.Exponential(var=dvar, lscale=lscale)
                    ks_model.add_item(item, kernel=k)

    def observe(self, *, t, white, black, winner):
        if winner == "white":
            self.ks_model.observe([white], [black], t=t/YEAR)
        elif winner == "black":
            self.ks_model.observe([black], [white], t=t/YEAR)
        else:  # winner == "tie":
            self.ks_model.observe([white], [black], t=t/YEAR, tie=True)

    def evaluate_obs(self, *, t, white, black, winner):
        probs = self.ks_model.probabilities([white], [black], t=year)
        if winner == "white":
            outcome = 0
        elif winner == "black":
            outcome = 2
        else:  # winner == "tie":
            outcome = 1
        log_loss = -log(probs[outcome])
        accuracy = 1.0 if outcome == np.argmax(probs) else 0.0
        return log_loss, accuracy


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
            for item in (obs["white"], obs["black"]):
                if item not in ks_model.item:
                    year = datetime(datetime.fromtimestamp(
                            obs["t"], timezone.utc).year, 1, 1).timestamp()
                    ks_model.add_item(item, kernel=ks.kernel.Wiener(
                            var=wvar/YEAR, t0=year, var_t0=cvar))

    def observe(self, *, t, white, black, winner):
        year = datetime(datetime.fromtimestamp(
                t, timezone.utc).year, 1, 1).timestamp()
        if winner == "white":
            self.ks_model.observe([white], [black], t=year)
        elif winner == "black":
            self.ks_model.observe([black], [white], t=year)
        else:  # winner == "tie":
            self.ks_model.observe([white], [black], t=year, tie=True)

    def evaluate_obs(self, *, t, white, black, winner):
        year = datetime(datetime.fromtimestamp(
                t, timezone.utc).year, 1, 1).timestamp()
        probs = self.ks_model.probabilities([white], [black], t=year)
        if winner == "white":
            outcome = 0
        elif winner == "black":
            outcome = 2
        else:  # winner == "tie":
            outcome = 1
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
            p1, pt, p2 = self.bl_model.predict(obs["white"], obs["black"])
            if obs["winner"] == "white":
                self._loglike += log(p1)
                self.bl_model.observe(obs["white"], obs["black"])
            elif obs["winner"] == "black":
                self._loglike += log(p2)
                self.bl_model.observe(obs["black"], obs["white"])
            else:  # obs["winner"] == "tie":
                self._loglike += log(pt)
                self.bl_model.observe(obs["white"], obs["black"], tie=True)

    def evaluate(self, *, begin):
        begin_ts = int(begin.timestamp())
        log_loss = 0
        accuracy = 0
        n_obs = 0
        for obs in iterate_dataset(DATASET):
            probs = self.bl_model.predict(obs["white"], obs["black"])
            if obs["t"] < begin_ts:
                continue
            if obs["winner"] == "white":
                outcome = 0
                self.bl_model.observe(obs["white"], obs["black"])
            elif obs["winner"] == "black":
                outcome = 2
                self.bl_model.observe(obs["black"], obs["white"])
            else:  # obs["winner"] == "tie":
                outcome = 1
                self.bl_model.observe(obs["white"], obs["black"], tie=True)
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


class CountModel(KickScoreModel):
    pass


class DifferenceModel(KickScoreModel):
    pass
