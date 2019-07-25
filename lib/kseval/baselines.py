import collections
import trueskill as ts

from math import exp, erfc, sqrt


class EloModel:

    def __init__(self, *, margin, lr):
        self.margin = margin
        self.lr = lr
        self.score = collections.defaultdict(lambda: 0)

    def observe(self, winner, loser, tie=False):
        pw, _, pl = self.predict(winner, loser)
        if tie:
            delta = self.lr * (pl - pw)
        else:
            delta = self.lr * (1.0 - pw)
        self.score[winner] += delta
        self.score[loser] -= delta

    def predict(self, a, b):
        pa = 1.0 / (1.0 + exp(-(self.score[a] - self.score[b] - self.margin)))
        pb = 1.0 / (1.0 + exp(-(self.score[b] - self.score[a] - self.margin)))
        return pa, 1.0 - pa - pb, pb


class TrueSkillModel:

    def __init__(self, *, margin, sigma, tau):
        self.env = ts.TrueSkill(
            # Initial mean of rating.
            mu=0.0,
            # Initial std. dev. of rating.
            sigma=sigma,
            # Scales the sigmoid function (denominator is sqrt(2) * beta).
            beta=1/sqrt(2.0),
            # Std. dev. of brownian dynamics.
            tau=tau,
            # Draw probability if skill difference is 0.
            draw_probability=(2 * ndtr(margin) - 1.0),
        )
        self.margin = margin
        self.rating = collections.defaultdict(self.env.create_rating)

    def observe(self, winner, loser, tie=False):
        self.rating[winner], self.rating[loser] = ts.rate_1vs1(
                self.rating[winner], self.rating[loser], drawn=tie,
                env=self.env)

    def predict(self, a, b):
        ma, sa = self.rating[a]
        mb, sb = self.rating[b]
        denom = sqrt(1.0 + sa*sa + sb*sb)
        pa = ndtr((ma - mb - self.margin) / denom)
        pb = ndtr((mb - ma - self.margin) / denom)
        return pa, 1.0 - pa - pb, pb


def ndtr(x):
    """Normal cumulative density function."""
    # If X ~ N(0,1), returns P(X < x).
    return erfc(-x / sqrt(2.0)) / 2.0
