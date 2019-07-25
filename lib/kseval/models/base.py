import abc
import json
import os

from datetime import datetime, timezone
from ..utils import data_path


YEAR = 365.25 * 24 * 60 * 60  # Number of seconds in a year.


class PredictiveModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fit(self, *, cutoff=None):
        """Fit the model.

        `cutoff` is an optional datetime object. Only observations up to
        `cutoff` (exclusive) are used to fit the model.
        """

    @abc.abstractmethod
    def evaluate(self, *, begin, end):
        """Evaluate the model on a test set.
        
        The model is evaluated on observations between `begin` (inclusive) and
        `end` (exclusive). Returns a triplet containing:

        1. number of observations
        2. sum of log-loss
        3. sum of accuracy
        """

    @property
    @abc.abstractmethod
    def log_likelihood(self):
        """Compute the marginal log-likelihood of the model."""

    @classmethod
    @abc.abstractmethod
    def get_dates(cls, begin=None):
        """Return dates of observations in the dataset."""


class KickScoreModel(PredictiveModel, metaclass=abc.ABCMeta):

    def __init__(self, ks_model, fit_params):
        self.ks_model = ks_model
        self.fit_params = fit_params

    @abc.abstractmethod
    def observe(self, **kwargs):
        """Add observation to the model."""

    @abc.abstractmethod
    def evaluate_obs(self, **kwargs):
        """Evaluate observation wit a fitted model."""

    def fit(self, *, cutoff=None):
        if cutoff is not None:
            cutoff_ts = int(cutoff.timestamp())
        else:
            cutoff_ts = float("inf")
        for obs in iterate_dataset(self.dataset):
            if obs["t"] >= cutoff_ts:
                break
            self.observe(**obs)
        converged = self.ks_model.fit(**self.fit_params)
        return converged

    def evaluate(self, *, begin, end):
        begin_ts = int(begin.timestamp())
        end_ts = int(end.timestamp())
        log_loss = 0
        accuracy = 0
        n_obs = 0
        for obs in iterate_dataset(self.dataset):
            if begin_ts <= obs["t"] < end_ts:
                ll, acc = self.evaluate_obs(**obs)
                log_loss += ll
                accuracy += acc
                n_obs += 1
        return n_obs, log_loss, accuracy

    @property
    def log_likelihood(self):
        return self.ks_model.log_likelihood

    @classmethod
    def get_dates(cls, begin=None):
        if begin is None:
            cutoff = float("-inf")
        else:
            cutoff = int(begin.timestamp())
        dates = set()
        for obs in iterate_dataset(cls.dataset):
            if obs["t"] >= cutoff:
                dates.add(datetime.fromtimestamp(
                        obs["t"], timezone.utc).date())
        return sorted(dates)


def iterate_dataset(fname):
    with open(data_path(fname)) as f:
        for line in f:
            yield json.loads(line.strip())
