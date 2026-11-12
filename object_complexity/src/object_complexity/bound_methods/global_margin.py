"""GlobalMargin: one margin for every query, learned from how far the estimate overshot on other categories.

For the held-out category, the other categories' errors (r_hat - true_recall) are sorted, and the margin is
the error that the target share of them fall below (split conformal: the ceil(coverage x (K + 1))-th
smallest of K errors). Y = r_hat - margin. By default there is one margin per specificity level.

Unlike the binomial bound, this corrects for the synthetic-to-real gap: if synthetic recall usually runs
high, the margin grows to match. It treats every query alike, so easy and hard queries get the same margin.
"""
import numpy as np

from .base import RecallBound


class GlobalMargin(RecallBound):
    name = "global_margin"

    @classmethod
    def add_method_arguments(cls, p):
        p.add_argument("--pool_specs", type=int, default=0,
                       help="global_margin: 1 = one margin from all specificity levels together; 0 = a margin per level")

    def method_params(self):
        return {"pool_specs": self.args.pool_specs}

    def fit(self, calib):
        err = calib.r_hat - calib.true_recall  # how far the estimate overshot, per calibration row
        if self.args.pool_specs:
            self.margins = {"all": _conformal_margin(err.to_numpy(), self.coverage)}
        else:
            self.margins = {s: _conformal_margin(e.to_numpy(), self.coverage) for s, e in err.groupby(calib.spec)}

    def lower_bound(self, rows):
        keys = ["all"] * len(rows) if self.args.pool_specs else list(rows.spec)
        return rows.r_hat.to_numpy() - np.array([self.margins[k] for k in keys])


def _conformal_margin(errors, coverage):
    """The ceil(coverage x (K + 1))-th smallest of K errors; infinite (bound = 0) if there are too few."""
    errors = np.sort(errors)
    rank = int(np.ceil(coverage * (len(errors) + 1)))
    return errors[rank - 1] if rank <= len(errors) else np.inf