import abc
import kickscore as ks
import numpy as np

from .base import PredictiveModel, KickScoreModel, iterate_dataset, YEAR
from .. import baselines

from datetime import datetime, timezone
from math import log


DATASET = "kdd-tennis.txt"


class BinaryModel(KickScoreModel, metaclass=abc.ABCMeta):

    dataset = DATASET

    def __init__(self, kernel, obs_type="probit", method="ep", max_iter=100, lr=1.0):
        ks_model = ks.BinaryModel(obs_type=obs_type)
        fit_params = {"method": method, "max_iter": max_iter, "lr": lr}
        self._kern = kernel
        super().__init__(ks_model, fit_params)

    def observe(self, *, t, winner, loser):
        for item in (winner, loser):
            if item not in self.ks_model.item:
                self.ks_model.add_item(item, kernel=self._kern)
        self.ks_model.observe([winner], [loser], t=t)

    def evaluate_obs(self, *, t, winner, loser):
        for item in (winner, loser):
            if item not in self.ks_model.item:
                self.ks_model.add_item(item, kernel=self._kern)
        pw, _ = self.ks_model.probabilities([winner], [loser], t=t)
        return -log(pw), 1.0 if pw > 0.5 else 0.0


class ConstantModel(BinaryModel):

    def __init__(self, *, cvar, method="ep", obs_type="probit"):
        kernel = ks.kernel.Constant(var=cvar)
        super().__init__(kernel, obs_type=obs_type, method=method)


class DynamicModel(BinaryModel):

    def __init__(self, *, nu, cvar, dvar, lscale,
            method="ep", obs_type="probit", max_iter=100):
        if nu == "1/2":
            dkern = ks.kernel.Exponential
        elif nu == "3/2":
            dkern = ks.kernel.Matern32
        elif nu == "5/2":
            dkern = ks.kernel.Matern52
        else:
            raise ValueError("invalid value for `nu`: {}".format(nu))
        kernel = (ks.kernel.Constant(var=cvar)
                + dkern(var=dvar, lscale=(lscale * YEAR)))
        super().__init__(kernel,
                obs_type=obs_type, method=method, max_iter=max_iter)


class WienerModel(BinaryModel):

    def __init__(self, *, cvar, wvar):
        super().__init__(None, obs_type="probit", method="ep")
        WienerModel.init_items(self.ks_model, cvar, wvar)

    @staticmethod
    def init_items(ks_model, cvar, wvar):
        for obs in iterate_dataset(DATASET):
            for item in (obs["winner"], obs["loser"]):
                if item not in ks_model.item:
                    ks_model.add_item(item, kernel=ks.kernel.Wiener(
                            var=wvar/YEAR, t0=obs["t"], var_t0=cvar))


class DoubleDynModel(BinaryModel):

    def __init__(self, *, cvar, lvar, llscale, svar, slscale,
            method="ep", obs_type="probit"):
        kernel = (ks.kernel.Constant(var=cvar)
                + ks.kernel.Matern52(var=lvar, lscale=(llscale * YEAR))
                + ks.kernel.Exponential(var=svar, lscale=(slscale * YEAR)))
        super().__init__(kernel, obs_type=obs_type, method=method)


class AffineDynamicModel(KickScoreModel):

    dataset = DATASET

    def __init__(self, *, ovar, svar, dvar, lscale):
        ks_model = ks.BinaryModel(obs_type="probit")
        AffineDynamicModel.init_items(ks_model, ovar, svar, dvar, lscale)
        fit_params = {"method": "ep", "max_iter": 100, "lr": 1.0}
        super().__init__(ks_model, fit_params)

    @staticmethod
    def init_items(ks_model, ovar, svar, dvar, lscale):
        for obs in iterate_dataset(DATASET):
            for item in (obs["winner"], obs["loser"]):
                if item not in ks_model.item:
                    k = ks.kernel.Affine(
                            var_offset=ovar, var_slope=svar, t0=obs["t"]/YEAR)
                    k += ks.kernel.Exponential(var=dvar, lscale=lscale)
                    ks_model.add_item(item, kernel=k)

    def observe(self, *, t, winner, loser):
        self.ks_model.observe([winner], [loser], t=t/YEAR)

    def evaluate_obs(self, *, t, winner, loser):
        pw, _ = self.ks_model.probabilities([winner], [loser], t=t/YEAR)
        return -log(pw), 1.0 if pw > 0.5 else 0.0


class AffineWienerModel(KickScoreModel):

    dataset = DATASET

    def __init__(self, *, ovar, svar, wvar, method="ep"):
        ks_model = ks.BinaryModel(obs_type="probit")
        AffineWienerModel.init_items(ks_model, ovar, svar, wvar)
        fit_params = {"method": method, "max_iter": 100, "lr": 1.0}
        super().__init__(ks_model, fit_params)

    @staticmethod
    def init_items(ks_model, ovar, svar, wvar):
        for obs in iterate_dataset(DATASET):
            for item in (obs["winner"], obs["loser"]):
                if item not in ks_model.item:
                    k = ks.kernel.Affine(
                            var_offset=ovar, var_slope=svar, t0=obs["t"]/YEAR)
                    k += ks.kernel.Wiener(
                            var=wvar, t0=obs["t"]/YEAR, var_t0=0.0001)
                    ks_model.add_item(item, kernel=k)

    def observe(self, *, t, winner, loser):
        self.ks_model.observe([winner], [loser], t=t/YEAR)

    def evaluate_obs(self, *, t, winner, loser):
        pw, _ = self.ks_model.probabilities([winner], [loser], t=t/YEAR)
        return -log(pw), 1.0 if pw > 0.5 else 0.0


class PeriodicModel(KickScoreModel):

    dataset = DATASET

    def __init__(self, *, cvar, dvar, dlscale, pvar, plscale):
        self._kern = (ks.kernel.Constant(var=cvar)
                + ks.kernel.Exponential(var=dvar, lscale=(dlscale * YEAR))
                + ks.kernel.PeriodicExponential(
                        var=pvar, lscale=(plscale * YEAR), period=YEAR))
        ks_model = ks.BinaryModel(obs_type="probit")
        fit_params = {"method": "ep", "max_iter": 100, "lr": 1.0}
        super().__init__(ks_model, fit_params)

    def observe(self, *, t, winner, loser):
        for item in (winner, loser):
            if item not in self.ks_model.item:
                self.ks_model.add_item(item, kernel=self._kern, fitter="batch")
        self.ks_model.observe([winner], [loser], t=t)

    def evaluate_obs(self, *, t, winner, loser):
        for item in (winner, loser):
            if item not in self.ks_model.item:
                self.ks_model.add_item(item, kernel=self._kern, fitter="batch")
        pw, _ = self.ks_model.probabilities([winner], [loser], t=t)
        return -log(pw), 1.0 if pw > 0.5 else 0.0


class TTTModel(KickScoreModel):

    dataset = DATASET

    def __init__(self, *, cvar, wvar):
        ks_model = ks.BinaryModel(obs_type="probit")
        TTTModel.init_items(ks_model, cvar, wvar)
        fit_params = {"method": "ep", "max_iter": 100, "lr": 1.0}
        super().__init__(ks_model, fit_params)

    @staticmethod
    def init_items(ks_model, cvar, wvar):
        for obs in iterate_dataset(DATASET):
            for item in (obs["winner"], obs["loser"]):
                if item not in ks_model.item:
                    year = datetime(datetime.fromtimestamp(
                            obs["t"], timezone.utc).year, 1, 1).timestamp()
                    ks_model.add_item(item, kernel=ks.kernel.Wiener(
                            var=wvar/YEAR, t0=year, var_t0=cvar))

    def observe(self, *, t, winner, loser):
        year = datetime(datetime.fromtimestamp(
                t, timezone.utc).year, 1, 1).timestamp()
        self.ks_model.observe([winner], [loser], t=year)

    def evaluate_obs(self, *, t, winner, loser):
        year = datetime(datetime.fromtimestamp(
                t, timezone.utc).year, 1, 1).timestamp()
        pw, _ = self.ks_model.probabilities([winner], [loser], t=year)
        return -log(pw), 1.0 if pw > 0.5 else 0.0


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
            pw, _, _ = self.bl_model.predict(obs["winner"], obs["loser"])
            self._loglike += log(pw)
            self.bl_model.observe(obs["winner"], obs["loser"])

    def evaluate(self, *, begin):
        begin_ts = int(begin.timestamp())
        log_loss = 0
        accuracy = 0
        n_obs = 0
        for obs in iterate_dataset(DATASET):
            if obs["t"] < begin_ts:
                continue
            pw, _, _ = self.bl_model.predict(obs["winner"], obs["loser"])
            log_loss += -log(pw)
            accuracy += 1.0 if pw > 0.5 else 0.0
            n_obs += 1
            self.bl_model.observe(obs["winner"], obs["loser"])
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
