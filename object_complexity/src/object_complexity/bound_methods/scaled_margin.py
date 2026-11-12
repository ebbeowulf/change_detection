"""ScaledMargin: a margin that adapts to each query, learned from the query's performance vector.

Two random forests, trained on the other categories, predict for each row:
  mu  expected overshoot (r_hat - true_recall) for a query like this one: shifts the bound
  s   typical size of that overshoot's variation for a query like this one: sets the width
The bound is Y = r_hat - mu - k * s. The multiplier k is set so the promise holds for the target share of
calibration cases: within fit(), each calibration category in turn is left out, the forests are trained on
the rest, and its scaled error (error - mu) / s is recorded (jackknife+). Every score used to set k therefore
comes from forests that never saw that category; the held-out category itself is never seen at all.

Inputs (--features, any of): rhat = r_hat and n; profile = prof_d*, prof_s*; size = log_size; acc = acc*.
The specificity level is not an input; r_hat and the threshold-dependent columns carry the operating point.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import RecallBound

GROUPS = {"rhat": ["r_hat", "n"], "size": ["log_size"]}  # profile and acc columns are found by prefix


class ScaledMargin(RecallBound):
    name = "scaled_margin"

    @classmethod
    def add_method_arguments(cls, p):
        p.add_argument("--features", type=str, default="rhat,profile,size",
                       help="scaled_margin: comma-separated input groups: rhat, profile, size, acc")
        p.add_argument("--trees", type=int, default=100, help="scaled_margin: trees per random forest")
        p.add_argument("--min_leaf", type=int, default=5, help="scaled_margin: minimum rows per forest leaf")
        p.add_argument("--min_spread", type=float, default=0.01, help="scaled_margin: floor on the predicted spread s")

    def method_params(self):
        a = self.args
        return {"features": a.features, "trees": a.trees, "min_leaf": a.min_leaf, "min_spread": a.min_spread}

    def columns(self, table):
        """Input columns for the chosen feature groups, in a fixed order."""
        cols = []
        for g in self.args.features.split(","):
            if g in GROUPS:
                cols += GROUPS[g]
            elif g == "profile":
                cols += sorted([c for c in table.columns if c.startswith("prof_")], key=_natural)
            elif g == "acc":
                cols += sorted([c for c in table.columns if c.startswith("acc")], key=_natural)
            else:
                raise ValueError(f"unknown feature group: {g}")
        return cols

    def _forests(self, X, e):
        """Train mu on the error, then s on how far each error sits from mu (using out-of-bag mu, so s is honest)."""
        kw = dict(n_estimators=self.args.trees, min_samples_leaf=self.args.min_leaf, n_jobs=-1, random_state=0)
        mu = RandomForestRegressor(oob_score=True, **kw).fit(X, e)
        mu_oob = np.where(np.isnan(mu.oob_prediction_), e.mean(), mu.oob_prediction_)
        s = RandomForestRegressor(**kw).fit(X, np.abs(e - mu_oob))
        return mu, s

    def _predict(self, forests, X):
        mu, s = forests
        return mu.predict(X), np.maximum(s.predict(X), self.args.min_spread)

    def fit(self, calib):
        self.cols = self.columns(calib)
        X = calib[self.cols].to_numpy()
        e = (calib.r_hat - calib.true_recall).to_numpy()  # how far the estimate overshot
        cats = calib.category.to_numpy()

        # Scaled errors for each calibration category, from forests trained without it (jackknife+).
        scores = []
        for c in np.unique(cats):
            out = cats == c
            mu, s = self._predict(self._forests(X[~out], e[~out]), X[out])
            scores.append((e[out] - mu) / s)
        scores = np.sort(np.concatenate(scores))
        rank = int(np.ceil(self.coverage * (len(scores) + 1)))
        self.k = scores[rank - 1] if rank <= len(scores) else np.inf

        self.final = self._forests(X, e)  # trained on all calibration categories, for the held-out one

    def lower_bound(self, rows):
        mu, s = self._predict(self.final, rows[self.cols].to_numpy())
        return rows.r_hat.to_numpy() - mu - self.k * s


def _natural(name):
    """Sort prof_d2 before prof_d10."""
    head = name.rstrip("0123456789")
    return (head, int(name[len(head):]))