import abc
import kickscore as ks
import numpy as np

from .base import PredictiveModel, iterate_dataset

from math import log


SETTINGS = {
    "wol": {
        "path": "kdd-starcraft-wol.txt",
        "idx1": 30828,
        "idx2": 43159,
    },
    "hots": {
        "path": "kdd-starcraft-hots.txt",
        "idx1": 14291,
        "idx2": 20007,
    },
}


class StarCraftModel(PredictiveModel, metaclass=abc.ABCMeta):

    def __init__(self, dataset, obs_type="probit",
            method="ep", max_iter=100, lr=1.0):
        self.ks_model = ks.BinaryModel(obs_type=obs_type)
        self.fit_params = {"method": method, "max_iter": max_iter, "lr": lr}
        self.dataset = dataset

    def fit(self, *, cutoff=None):
        path = SETTINGS[self.dataset]["path"]
        idx1 = SETTINGS[self.dataset]["idx1"]
        for i, obs in enumerate(iterate_dataset(path)):
            if i == idx1:
                break
            feats1, feats2 = self.make_features(obs["winner"], obs["loser"])
            self.ks_model.observe(winners=feats1, losers=feats2, t=0.0)
        converged = self.ks_model.fit(**self.fit_params)
        return converged

    def evaluate(self):
        path = SETTINGS[self.dataset]["path"]
        idx2 = SETTINGS[self.dataset]["idx2"]
        log_loss = 0
        accuracy = 0
        n_obs = 0
        for i, obs in enumerate(iterate_dataset(path)):
            if i >= idx2:
                feats1, feats2 = self.make_features(obs["winner"], obs["loser"])
                pw, _ = self.ks_model.probabilities(feats1, feats2, t=0.0)
                log_loss += -log(pw)
                accuracy += 1.0 if pw > 0.5 else 0.0 
                n_obs += 1
        return n_obs, log_loss, accuracy

    @abc.abstractmethod
    def make_features(self, winner, loser):
        """Make feature vectors for the two players."""

    @property
    def log_likelihood(self):
        path = SETTINGS[self.dataset]["path"]
        idx1 = SETTINGS[self.dataset]["idx1"]
        idx2 = SETTINGS[self.dataset]["idx2"]
        loglike = 0
        for i, obs in enumerate(iterate_dataset(path)):
            if idx1 <= i < idx2:
                feats1, feats2 = self.make_features(obs["winner"], obs["loser"])
                pw, _ = self.ks_model.probabilities(feats1, feats2, t=0.0)
                loglike += log(pw)
        return loglike

    @classmethod
    def get_dates(cls, begin=None):
        raise NotImplementedError()


class ConstantModel(StarCraftModel):

    def __init__(self, *, dataset, cvar):
        super().__init__(dataset)
        self._ckern = ks.kernel.Constant(var=cvar)

    def make_features(self, winner, loser):
        for item in (winner, loser):
            if item not in self.ks_model.item:
                self.ks_model.add_item(item, kernel=self._ckern)
        return [winner], [loser]


class IntransitiveModel(StarCraftModel):

    def __init__(self, *, dataset, cvar, xvar):
        super().__init__(dataset, lr=0.8)
        self._ckern = ks.kernel.Constant(var=cvar)
        self._xkern = ks.kernel.Constant(var=xvar)

    def make_features(self, winner, loser):
        for item in (winner, loser):
            if item not in self.ks_model.item:
                self.ks_model.add_item(item, kernel=self._ckern)
        x = "{}-{}".format(*sorted((winner, loser)))
        if x not in self.ks_model.item:
            self.ks_model.add_item(x, kernel=self._xkern)
        if winner < loser:
            return [winner, x], [loser]
        else:
            return [winner], [loser, x]
