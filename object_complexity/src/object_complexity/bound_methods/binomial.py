"""BinomialBound: Clopper-Pearson lower bound from the k of n synthetic examples detected.

Uses no other categories. It treats the synthetic examples as a random sample of the real target, so it
covers only the luck of which examples were drawn, not systematic differences between synthetic and real.
"""
import numpy as np
from scipy.stats import beta

from .base import RecallBound


class BinomialBound(RecallBound):
    name = "binomial"

    def lower_bound(self, rows):
        k, n = rows.k.to_numpy(), rows.n.to_numpy()
        alpha = 1 - self.coverage  # e.g. 0.1: the bound is too high 1 time in 10, if the sample is representative
        return np.where(k > 0, beta.ppf(alpha, np.maximum(k, 1), n - k + 1), 0.0)